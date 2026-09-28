#!/usr/bin/env python3
"""
Read-only dataset analysis for the Amazon ML Challenge 2026 (Business Entity Resolution).

Prints AGGREGATE statistics only (counts, shares, length percentiles, top tokens). It
never prints raw records in bulk and never modifies the TSV files. Standard library only.

Sections:
  1. Files:        presence, sizes, headers
  2. Sources:      rows, missing values, duplicate IDs, ID prefixes, countries, lengths, patterns
  3. Ground truth: match-count distribution, S2/S3 mix, whether an S2/S3 record maps to
                   several S1 records (the one-match assumption), coverage of source IDs
  4. Noise:        how true pairs differ (casing, punctuation, abbreviations, exact/near
                   matches, postcodes), with random non-matching pairs as a baseline
  5. Blocking:     how often true pairs share cheap blocking keys, and brute-force pair counts

Usage (from the repo root):
    python3 code/business_entity_resolution/src/analyze_dataset.py \
        [--data-dir student_resource/dataset] [--sample 20000] [--json-out report.json]
"""

import argparse
import collections
import difflib
import json
import os
import random
import re
import statistics
import sys
import unicodedata

EXPECTED = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Tokens used only to MEASURE how often abbreviations and variants occur (not a model).
ABBREV_PAIRS = {
    "name": [("pvt", "private"), ("ltd", "limited"), ("corp", "corporation"), ("inc", "incorporated"),
             ("co", "company"), ("&", "and"), ("intl", "international"), ("mfg", "manufacturing")],
    "address": [("rd", "road"), ("st", "street"), ("ave", "avenue"), ("blvd", "boulevard"),
                ("dr", "drive"), ("ln", "lane"), ("nr", "near"), ("opp", "opposite"), ("apt", "apartment")],
}
LEGAL_SUFFIXES = ["pvt", "private", "ltd", "limited", "llc", "llp", "inc", "corp", "corporation", "co",
                  "company", "plc", "sa", "sas", "sarl", "gmbh", "lp"]
POSTCODE_RE = re.compile(r"(?<!\d)(\d{3}\s?\d{3}|\d{5}(?:-\d{4})?)(?!\d)")
TOKEN_RE = re.compile(r"[a-z0-9]+|&")


# --------------------------------------------------------------------------- IO

def read_tsv(path):
    """Yield (header, row) for a TSV. Uses a plain split on tabs, matching the official format."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        yield header, None
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                yield header, line.split("\t")


def load_source(path, keep_records):
    """Load one source file; returns (stats, records-dict-or-None)."""
    rows = 0
    header = None
    bad_width = 0
    missing = collections.Counter()
    ids = set()
    dup_ids = 0
    prefixes = collections.Counter()
    countries = collections.Counter()
    name_len, addr_len, name_tok, addr_tok = [], [], [], []
    pattern = collections.Counter()
    legal = collections.Counter()
    records = {} if keep_records else None
    for hdr, row in read_tsv(path):
        if row is None:
            header = hdr
            continue
        rows += 1
        if len(row) != len(header):
            bad_width += 1
            row = (row + [""] * len(header))[: len(header)]
        rec = dict(zip(header, row))
        for col in header:
            if not rec[col].strip():
                missing[col] += 1
        eid = rec.get("entity_id", "").strip()
        if eid in ids:
            dup_ids += 1
        ids.add(eid)
        prefixes[eid[:3]] += 1
        countries[rec.get("country", "").strip()] += 1
        name, addr = rec.get("business_name", ""), rec.get("business_address", "")
        name_len.append(len(name))
        addr_len.append(len(addr))
        name_tok.append(len(name.split()))
        addr_tok.append(len(addr.split()))
        _patterns(name, addr, pattern)
        for t in set(tokens(name)) & set(LEGAL_SUFFIXES):
            legal[t] += 1
        if keep_records:
            records[eid] = (name, addr, rec.get("country", "").strip())
    stats = {
        "rows": rows,
        "columns": header,
        "columns_as_expected": header == SOURCE_COLUMNS,
        "rows_with_wrong_column_count": bad_width,
        "missing_or_empty": {c: missing[c] for c in header},
        "duplicate_ids": dup_ids,
        "id_prefixes": dict(prefixes),
        "countries": dict(countries.most_common()),
        "name_chars": _describe(name_len),
        "address_chars": _describe(addr_len),
        "name_tokens": _describe(name_tok),
        "address_tokens": _describe(addr_tok),
        "patterns_share": {k: round(v / rows, 4) for k, v in sorted(pattern.items())} if rows else {},
        "legal_suffix_share": {k: round(v / rows, 4) for k, v in legal.most_common(12)} if rows else {},
    }
    return stats, records, ids


def _patterns(name, addr, counter):
    """Count simple formatting patterns (shares are computed by the caller)."""
    for label, text in (("name", name), ("address", addr)):
        if not text.strip():
            continue
        if text.isupper():
            counter[f"{label}_all_upper"] += 1
        elif text.islower():
            counter[f"{label}_all_lower"] += 1
        if any(ord(ch) > 127 for ch in text):
            counter[f"{label}_non_ascii"] += 1
        if re.search(r"[^\w\s]", text):
            counter[f"{label}_has_punctuation"] += 1
        if re.search(r"\d", text):
            counter[f"{label}_has_digit"] += 1
    if POSTCODE_RE.search(addr):
        counter["address_has_postcode_like"] += 1
    if re.search(r"\b(near|nr|opp|opposite|behind|beside)\b", addr.lower()):
        counter["address_landmark_word"] += 1
    if "&" in name:
        counter["name_has_ampersand"] += 1


# --------------------------------------------------------------------------- text helpers

def fold(text):
    """Lowercase and strip accents (used only for measurement)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokens(text):
    return TOKEN_RE.findall(fold(text))


def squash(text):
    return " ".join(tokens(text))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def postcode(text):
    m = POSTCODE_RE.search(text)
    return m.group(1).replace(" ", "")[:5] if m else None


def _describe(values):
    if not values:
        return {}
    s = sorted(values)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    return {"min": s[0], "p05": q(0.05), "median": q(0.5), "mean": round(statistics.fmean(s), 2),
            "p95": q(0.95), "max": s[-1]}


def _share(n, d):
    return round(n / d, 4) if d else None


# --------------------------------------------------------------------------- ground truth

def analyze_ground_truth(path, s1_ids, s2_ids, s3_ids):
    truth = {}
    dup_rows = 0
    for hdr, row in read_tsv(path):
        if row is None:
            header = hdr
            continue
        s1 = row[0].strip()
        ids = [x.strip() for x in (row[1] if len(row) > 1 else "").strip().strip('"').split(",") if x.strip()]
        if s1 in truth:
            dup_rows += 1
        truth[s1] = ids
    counts = [len(v) for v in truth.values()]
    n = len(counts)
    dist = collections.Counter(min(c, 10) for c in counts)
    target_to_s1 = collections.defaultdict(set)
    within_dupes = 0
    for s1, ids in truth.items():
        within_dupes += len(ids) != len(set(ids))
        for t in ids:
            target_to_s1[t].add(s1)
    multi_owner = {t: len(s) for t, s in target_to_s1.items() if len(s) > 1}
    all_targets = [t for ids in truth.values() for t in ids]
    gt_s1 = set(truth)
    result = {
        "header": header,
        "rows": n,
        "duplicate_s1_rows": dup_rows,
        "rows_with_repeated_id_in_list": within_dupes,
        "s1_zero_matches": dist.get(0, 0),
        "s1_one_match": dist.get(1, 0),
        "s1_multiple_matches": sum(v for k, v in dist.items() if k >= 2),
        "share_zero": _share(dist.get(0, 0), n),
        "share_one": _share(dist.get(1, 0), n),
        "share_multiple": _share(sum(v for k, v in dist.items() if k >= 2), n),
        "matches_per_s1_distribution(10=10+)": dict(sorted(dist.items())),
        "matches_per_s1_mean": round(statistics.fmean(counts), 4) if counts else None,
        "matches_per_s1_mean_excluding_zero": round(statistics.fmean([c for c in counts if c]), 4)
        if any(counts) else None,
        "matches_per_s1_median": statistics.median(counts) if counts else None,
        "matches_per_s1_max": max(counts) if counts else None,
        "true_pairs_total": len(all_targets),
        "true_pairs_to_S2": sum(t.startswith("S2-") for t in all_targets),
        "true_pairs_to_S3": sum(t.startswith("S3-") for t in all_targets),
        "s1_matching_both_S2_and_S3": sum(
            1 for ids in truth.values() if any(t.startswith("S2-") for t in ids) and any(t.startswith("S3-") for t in ids)),
        "ONE_MATCH_ASSUMPTION_each_S2S3_has_at_most_one_S1": not multi_owner,
        "s2s3_records_matched_to_multiple_s1": len(multi_owner),
        "max_s1_per_s2s3_record": max(multi_owner.values()) if multi_owner else 1,
        "gt_s1_not_in_source1": len(gt_s1 - s1_ids),
        "source1_not_in_gt": len(s1_ids - gt_s1),
        "gt_targets_not_in_sources": len(set(all_targets) - s2_ids - s3_ids),
        "gt_targets_with_bad_prefix": sum(not t.startswith(("S2-", "S3-")) for t in all_targets),
        "S2_records_never_matched": len(s2_ids - set(target_to_s1)),
        "S3_records_never_matched": len(s3_ids - set(target_to_s1)),
        "share_S2_matched": _share(len(s2_ids & set(target_to_s1)), len(s2_ids)),
        "share_S3_matched": _share(len(s3_ids & set(target_to_s1)), len(s3_ids)),
    }
    return result, truth


# --------------------------------------------------------------------------- pair noise

def pair_features(a, b):
    """Measure how two records differ (for statistics only)."""
    (na, aa, ca), (nb, ab, cb) = a, b
    ta, tb = tokens(na), tokens(nb)
    xa, xb = tokens(aa), tokens(ab)
    pa, pb = postcode(aa), postcode(ab)
    return {
        "name_exact": na == nb,
        "name_exact_casefold": na.casefold() == nb.casefold(),
        "name_exact_normalized": squash(na) == squash(nb),
        "name_same_token_set": set(ta) == set(tb),
        "name_case_differs_only": na != nb and na.casefold() == nb.casefold(),
        "name_punct_differs": squash(na) == squash(nb) and na.casefold() != nb.casefold(),
        "name_ratio": difflib.SequenceMatcher(None, squash(na), squash(nb)).ratio(),
        "name_jaccard": jaccard(ta, tb),
        "address_exact": aa == ab,
        "address_exact_normalized": squash(aa) == squash(ab),
        "address_ratio": difflib.SequenceMatcher(None, squash(aa), squash(ab)).ratio(),
        "address_jaccard": jaccard(xa, xb),
        "same_country": ca == cb,
        "postcode_both": pa is not None and pb is not None,
        "postcode_equal": pa is not None and pa == pb,
        "share_any_name_token": bool(set(ta) & set(tb)),
        "share_first_name_token": bool(ta and tb and ta[0] == tb[0]),
        "share_any_address_token": bool(set(xa) & set(xb)),
        "share_name_or_postcode": bool(set(ta) & set(tb)) or (pa is not None and pa == pb),
    }


def summarize_pairs(feats):
    out = {}
    if not feats:
        return out
    for key in feats[0]:
        vals = [f[key] for f in feats]
        if isinstance(vals[0], bool):
            out[key] = _share(sum(vals), len(vals))
        else:
            s = sorted(vals)
            out[key] = {"p10": round(s[len(s) // 10], 3), "median": round(s[len(s) // 2], 3),
                        "mean": round(statistics.fmean(s), 3), "share_ge_0.9": _share(sum(v >= 0.9 for v in s), len(s))}
    out["n_pairs"] = len(feats)
    postcode_both = sum(f["postcode_both"] for f in feats)
    out["postcode_equal_given_both_present"] = _share(sum(f["postcode_equal"] for f in feats), postcode_both)
    return out


def abbreviation_usage(records_list):
    """Share of records using the short vs long form of common abbreviations."""
    out = {}
    for field_idx, kind in ((0, "name"), (1, "address")):
        counts = collections.Counter()
        for records in records_list:
            for rec in records.values():
                toks = set(tokens(rec[field_idx]))
                if kind == "name" and "&" in rec[0]:
                    toks.add("&")
                for short, long in ABBREV_PAIRS[kind]:
                    counts[short] += short in toks
                    counts[long] += long in toks
        total = sum(len(r) for r in records_list)
        out[kind] = {f"{s}/{l}": [_share(counts[s], total), _share(counts[l], total)] for s, l in ABBREV_PAIRS[kind]}
    return out


def non_ascii_by_country(records_list):
    tot, non = collections.Counter(), collections.Counter()
    for records in records_list:
        for name, addr, country in records.values():
            tot[country] += 1
            non[country] += any(ord(ch) > 127 for ch in name + addr)
    return {c: _share(non[c], tot[c]) for c in tot}


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only aggregate analysis of the challenge dataset.")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--sample", type=int, default=20000, help="max true/random pairs used for noise stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", help="optional path to also write the report as JSON")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    report = {}

    # 1. Files
    files = {}
    for split, names in EXPECTED.items():
        for name in names:
            p = os.path.join(args.data_dir, split, name)
            files[f"{split}/{name}"] = {"exists": os.path.isfile(p),
                                        "bytes": os.path.getsize(p) if os.path.isfile(p) else None}
    report["files"] = files
    missing = [k for k, v in files.items() if not v["exists"]]
    if missing:
        print(json.dumps(report, indent=2))
        print(f"ERROR: missing files under {args.data_dir}: {missing}", file=sys.stderr)
        return 1

    # 2. Sources (train kept in memory for pair analysis; test streamed)
    sources, train_records, ids = {}, {}, {}
    for split in ("train", "test"):
        for k in (1, 2, 3):
            key = f"{split}_source{k}"
            path = os.path.join(args.data_dir, split, f"{key}.tsv")
            stats, recs, id_set = load_source(path, keep_records=(split == "train"))
            sources[key] = stats
            ids[key] = id_set
            if recs is not None:
                train_records[k] = recs
            print(f"[loaded] {key}: {stats['rows']} rows", file=sys.stderr)
    report["sources"] = sources
    report["brute_force_pairs"] = {
        split: sources[f"{split}_source1"]["rows"]
        * (sources[f"{split}_source2"]["rows"] + sources[f"{split}_source3"]["rows"])
        for split in ("train", "test")
    }
    report["countries_only_in_test"] = sorted(
        set().union(*(sources[f"test_source{k}"]["countries"] for k in (1, 2, 3)))
        - set().union(*(sources[f"train_source{k}"]["countries"] for k in (1, 2, 3))))

    # 3. Ground truth
    gt, truth = analyze_ground_truth(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"),
                                     ids["train_source1"], ids["train_source2"], ids["train_source3"])
    report["ground_truth"] = gt

    # 4. Noise on true pairs vs random non-pairs
    true_pairs = [(s1, t) for s1, ts in truth.items() for t in ts]
    rng.shuffle(true_pairs)
    s1r, s2r, s3r = train_records[1], train_records[2], train_records[3]
    lookup = lambda t: (s2r if t.startswith("S2-") else s3r).get(t)  # noqa: E731
    scored = [(b, pair_features(s1r[a], lookup(b))) for a, b in true_pairs[: args.sample]
              if a in s1r and lookup(b) is not None]
    feats_true = [f for _, f in scored]
    feats_true_s2 = [f for b, f in scored if b.startswith("S2-")]
    feats_true_s3 = [f for b, f in scored if b.startswith("S3-")]
    true_set = {(a, b) for a, b in true_pairs}
    s1_keys, tgt_keys = list(s1r), list(s2r) + list(s3r)
    rand_pairs = []
    while len(rand_pairs) < min(args.sample, len(true_pairs) or args.sample):
        a, b = rng.choice(s1_keys), rng.choice(tgt_keys)
        if (a, b) not in true_set:
            rand_pairs.append((a, b))
    feats_rand = [pair_features(s1r[a], lookup(b)) for a, b in rand_pairs]
    report["true_pairs"] = summarize_pairs(feats_true)
    report["true_pairs_S1_S2"] = summarize_pairs(feats_true_s2)
    report["true_pairs_S1_S3"] = summarize_pairs(feats_true_s3)
    report["random_non_pairs"] = summarize_pairs(feats_rand)
    report["abbreviation_usage_train(short_share,long_share)"] = abbreviation_usage([s1r, s2r, s3r])
    report["non_ascii_share_by_country_train"] = non_ascii_by_country([s1r, s2r, s3r])

    # 5. Duplicate-looking names inside each source (normalized name + country)
    dup_names = {}
    for k, recs in train_records.items():
        c = collections.Counter((squash(n), cty) for n, _, cty in recs.values())
        dup_names[f"train_source{k}"] = _share(sum(v for v in c.values() if v > 1), len(recs))
    report["share_records_with_duplicate_normalized_name_in_same_source"] = dup_names

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
