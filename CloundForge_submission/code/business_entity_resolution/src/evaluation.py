#!/usr/bin/env python3
"""
Local macro F0.5 evaluator for the Amazon ML Challenge 2026 (Business Entity Resolution).

Reproduces the official metric (student_resource/README.md, "Evaluation Criteria"):

* For each Source 1 entity, compare the predicted set P with the true set T of
  S2/S3 IDs.
    - T empty and P empty                -> 1.0   (correct singleton)
    - T empty and P non-empty            -> 0.0   (false merge on a singleton)
    - T non-empty and P empty            -> 0.0   (recall 0)
    - otherwise precision = |P∩T|/|P|, recall = |P∩T|/|T| and
      F0.5 = 1.25·P·R / (0.25·P + R), which is 0.0 when |P∩T| = 0
* Macro-average the per-entity scores over ALL evaluated S1 entities.

Optionally scores a candidate file (candidate_pairs.tsv) too, and reports the blocking
recall ceiling, the candidate-set size, and the "oracle" macro F0.5: the best score any
matcher could reach if it kept exactly the true matches among the candidates.

Standard library only; independent of any model code. Usage (from the repo root):

    python3 code/business_entity_resolution/src/evaluation.py \
        --predictions path/to/matching_results.tsv \
        --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
        [--candidates path/to/candidate_pairs.tsv] \
        [--only-predicted] [--json]

By default every S1 entity in the ground truth is evaluated, and an entity that is
missing from the predictions counts as an empty prediction (a warning is printed).
Use --only-predicted to evaluate just the S1 entities present in the predictions file,
e.g. a held-out validation fold.
"""

import argparse
import json
import statistics
import sys

BETA = 0.5
_B2 = BETA * BETA


def f_beta(n_pred, n_true, n_hit):
    """Per-entity F0.5 from set sizes, following the official singleton rules."""
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or n_hit == 0:
        return 0.0
    precision = n_hit / n_pred
    recall = n_hit / n_true
    return (1 + _B2) * precision * recall / (_B2 * precision + recall)


def parse_id_list(field):
    """Split a comma-separated ID list, tolerating quotes, spaces and trailing commas."""
    field = field.strip().strip('"').strip("'")
    return [x.strip() for x in field.split(",") if x.strip()]


def read_id_list_tsv(path, warnings, label):
    """Read a two-column ``source1_entity_id<TAB>id,id,...`` file into {s1: set(ids)}.

    Skips the header row and blank lines, and handles a UTF-8 BOM and CRLF line endings.
    Problems (a duplicate S1 row, a repeated ID, a row without a tab) are appended to
    ``warnings`` rather than raised, so a malformed file can still be diagnosed.
    """
    mapping = {}
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
        if "\t" not in header:
            warnings.append(f"{label}: header has no TAB ({header.strip()[:80]!r}); is it a CSV?")
        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            s1, tab, rest = line.partition("\t")
            s1 = s1.strip()
            if not tab:
                warnings.append(f"{label}: line {line_num} has no TAB; treated as an empty list")
            ids = parse_id_list(rest)
            if len(ids) != len(set(ids)):
                warnings.append(f"{label}: repeated ID in the list for {s1}")
            if s1 in mapping:
                warnings.append(f"{label}: duplicate row for {s1}; lists merged")
                mapping[s1] |= set(ids)
            else:
                mapping[s1] = set(ids)
    return mapping


def evaluate(pred, truth, candidates=None, only_predicted=False):
    """Score predictions against the ground truth. Returns a dict of metrics.

    ``pred`` / ``truth`` / ``candidates`` map S1 id -> set of S2/S3 ids.
    """
    warnings = []
    if only_predicted:
        s1_ids = [s for s in truth if s in pred]
        unknown = [s for s in pred if s not in truth]
        if unknown:
            warnings.append(f"{len(unknown)} predicted S1 IDs are not in the ground truth; ignored")
    else:
        s1_ids = list(truth)
        missing = [s for s in s1_ids if s not in pred]
        if missing:
            warnings.append(
                f"{len(missing)} ground-truth S1 IDs have no prediction row; scored as empty "
                "(the official validator would reject such a file)"
            )
    if not s1_ids:
        raise ValueError("No S1 entities to evaluate: check the files and --only-predicted")

    scores, precisions, recalls = [], [], []
    single_scores, multi_scores = [], []
    tp_total = pred_total = true_total = exact = 0
    for s1 in s1_ids:
        t = truth[s1]
        p = pred.get(s1, set())
        hit = len(p & t)
        score = f_beta(len(p), len(t), hit)
        scores.append(score)
        (single_scores if not t else multi_scores).append(score)
        exact += p == t
        tp_total += hit
        pred_total += len(p)
        true_total += len(t)
        if p:
            precisions.append(hit / len(p))
        if t:
            recalls.append(hit / len(t))

    n = len(s1_ids)
    result = {
        "n_entities": n,
        "macro_f0.5": sum(scores) / n,
        "exact_set_match_rate": exact / n,
        "singletons": {
            "count": len(single_scores),
            "share": len(single_scores) / n,
            "mean_f0.5": _mean(single_scores),
            "predicted_empty": int(sum(single_scores)),  # a singleton scores 1.0 iff it was predicted empty
        },
        "with_matches": {
            "count": len(multi_scores),
            "mean_f0.5": _mean(multi_scores),
        },
        "micro_precision": tp_total / pred_total if pred_total else None,
        "micro_recall": tp_total / true_total if true_total else None,
        "mean_precision_where_predicted": _mean(precisions),
        "mean_recall_where_true": _mean(recalls),
        "predicted_pairs": pred_total,
        "true_pairs": true_total,
    }

    if candidates is not None:
        result["candidates"] = _evaluate_candidates(s1_ids, truth, pred, candidates, warnings)
    result["warnings"] = warnings
    return result


def _evaluate_candidates(s1_ids, truth, pred, candidates, warnings):
    """Blocking diagnostics: recall ceiling, set size and oracle score."""
    sizes, oracle = [], []
    covered = true_total = leaked = 0
    for s1 in s1_ids:
        t = truth[s1]
        c = candidates.get(s1, set())
        sizes.append(len(c))
        hit = len(c & t)
        covered += hit
        true_total += len(t)
        # Oracle: keep exactly the true matches that blocking recovered.
        oracle.append(f_beta(hit, len(t), hit))
        leaked += bool(pred.get(s1, set()) - c)
    if leaked:
        warnings.append(f"{leaked} S1 entities have predicted IDs that are not among their candidates")
    sizes_sorted = sorted(sizes)
    return {
        "pair_recall_ceiling": covered / true_total if true_total else None,
        "oracle_macro_f0.5": sum(oracle) / len(oracle),
        "total_candidate_pairs": sum(sizes),
        "mean_per_s1": _mean(sizes),
        "median_per_s1": statistics.median(sizes),
        "p95_per_s1": sizes_sorted[min(len(sizes_sorted) - 1, int(0.95 * len(sizes_sorted)))],
        "max_per_s1": sizes_sorted[-1],
        "s1_with_no_candidates": sum(1 for s in sizes if s == 0),
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def _format(result):
    lines = [
        f"Entities evaluated     : {result['n_entities']}",
        f"MACRO F0.5             : {result['macro_f0.5']:.5f}",
        f"  singletons           : {result['singletons']['count']} "
        f"({result['singletons']['share']:.1%}), mean F0.5 {_fmt(result['singletons']['mean_f0.5'])}",
        f"  with matches         : {result['with_matches']['count']}, "
        f"mean F0.5 {_fmt(result['with_matches']['mean_f0.5'])}",
        f"Exact set match rate   : {result['exact_set_match_rate']:.4f}",
        f"Micro precision/recall : {_fmt(result['micro_precision'])} / {_fmt(result['micro_recall'])}",
        f"Predicted / true pairs : {result['predicted_pairs']} / {result['true_pairs']}",
    ]
    cand = result.get("candidates")
    if cand:
        lines += [
            "Candidates:",
            f"  pair recall ceiling  : {_fmt(cand['pair_recall_ceiling'])}",
            f"  oracle macro F0.5    : {cand['oracle_macro_f0.5']:.5f}",
            f"  per S1 mean/med/p95/max : {cand['mean_per_s1']:.2f} / {cand['median_per_s1']} / "
            f"{cand['p95_per_s1']} / {cand['max_per_s1']}",
            f"  total pairs          : {cand['total_candidate_pairs']}",
        ]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    return "\n".join(lines)


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local macro F0.5 evaluator (Amazon ML Challenge 2026).")
    parser.add_argument("--predictions", "-p", required=True, help="matching_results.tsv to score")
    parser.add_argument("--ground-truth", "-g", required=True, help="train_ground_truth.tsv (or a fold of it)")
    parser.add_argument("--candidates", "-c", help="optional candidate_pairs.tsv for blocking diagnostics")
    parser.add_argument("--only-predicted", action="store_true",
                        help="evaluate only S1 entities present in the predictions file")
    parser.add_argument("--json", action="store_true", help="print the metrics as JSON")
    args = parser.parse_args(argv)

    warnings = []
    truth = read_id_list_tsv(args.ground_truth, warnings, "ground truth")
    pred = read_id_list_tsv(args.predictions, warnings, "predictions")
    cands = read_id_list_tsv(args.candidates, warnings, "candidates") if args.candidates else None
    result = evaluate(pred, truth, cands, only_predicted=args.only_predicted)
    result["warnings"] = warnings + result["warnings"]
    print(json.dumps(result, indent=2) if args.json else _format(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
