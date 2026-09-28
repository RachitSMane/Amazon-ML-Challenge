# business_entity_resolution

The team's solution for the Amazon ML Challenge 2026 (Business Entity Resolution).
This folder is what ships in the final zip as `code/business_entity_resolution/`
(see `student_resource/README.md`, "Final Submission Package").

> **Status: final.** The submitted `output/` files were produced by `ber.predict` (stage 1 +
> stage 2, see section 9 for the end-to-end reproduction commands).

## Layout

```
code/business_entity_resolution/
├── README.md             # this file: setup + exact run instructions
├── requirements.txt      # pinned dependencies (all MIT / Apache-2.0 / BSD-3; no GPL)
├── src/
│   ├── evaluation.py     # local macro F0.5 scorer + candidate-set diagnostics (stdlib only)
│   ├── analyze_dataset.py# read-only aggregate dataset analysis (EDA, stdlib only)
│   └── ber/
│       ├── config.py     # dataset discovery, expected files and row counts, cache paths
│       ├── io.py         # Arrow-backed loaders, raw Parquet cache, shared cache manifests
│       ├── normalize/
│       │   ├── text.py         # NFKC/casefold/punctuation, Latin accent folding, scripts
│       │   ├── legal_forms.py  # explicit legal-form lists (global, US, India, France, Indic)
│       │   ├── address.py      # placeholders, postcode/house/unit numbers, abbreviations
│       │   └── cache.py        # work/cache/norm_v1 builder, loaders and CLI
│       ├── tokens.py     # tokenization, per-pool/per-country vocab, DF/IDF, inverted CSR
│       ├── cv.py         # deterministic stratified S1 folds + 100k dev subset
│       ├── ranker.py     # stage-1 pair features + LightGBM candidate ranker
│       └── retrieval/
│           ├── sparse_topk.py  # chunked queries@inverted scoring, deterministic top-K
│           ├── channels.py     # C1 name_fold, C2 address, C3 name_stripped, K strong name
│           ├── union.py        # deduplicated union: summed score, channel mask, rank
│           ├── indic.py        # Indic romanization + phonetic skeleton key (experimental)
│           └── evaluate.py     # streaming pair/S1 recall and oracle F0.5
├── experiments/
│   ├── e02_retrieval.py  # dev-subset retrieval experiments -> work/experiments/retrieval
│   ├── e03_indic.py      # Indic channel experiment -> work/experiments/retrieval/indic
│   └── e03_ranker.py     # stage-1 ranker experiment -> work/experiments/ranker
└── tests/
    ├── test_evaluation.py
    ├── test_io.py               # fast loader tests on synthetic TSVs
    ├── test_io_dataset.py       # acceptance tests on the real dataset (skipped if absent)
    ├── test_normalize.py        # normalization rules and the normalized cache (synthetic)
    ├── test_normalize_dataset.py# normalized cache vs raw cache (skipped if not built)
    └── test_tokens.py  test_sparse_topk.py  test_retrieval.py  test_cv.py  test_ranker.py
```

All commands below run from the **repository root**, with Python ≥ 3.10.

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## 1. Dataset setup (local only, never committed)

The dataset is restricted challenge material. `.gitignore` blocks every `*.tsv`, `*.zip`
and `student_resource/dataset/`, so it cannot be committed by accident.

```bash
# 1. Inspect the official zip without extracting it
unzip -l /path/to/6ab10eb3b23ba_student_resource.zip

# 2. Extract to a temporary folder (the zip may contain macOS "__MACOSX/" and "._*" junk)
unzip -q /path/to/6ab10eb3b23ba_student_resource.zip -d /tmp/er_zip

# 3. Copy ONLY the dataset folder into the canonical location
mkdir -p student_resource/dataset
cp -R /tmp/er_zip/student_resource/dataset/. student_resource/dataset/   # adjust if the zip's top folder differs

# 4. Check: the analysis script lists all 7 expected files (exists/bytes) first
python3 code/business_entity_resolution/src/analyze_dataset.py --json-out /tmp/er_report.json
```

Expected result:

```
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## 2. Dataset analysis (read-only)

```bash
python3 code/business_entity_resolution/src/analyze_dataset.py \
    [--data-dir student_resource/dataset] [--sample 20000] [--json-out report.json]
```

It prints aggregate JSON only: row counts, missing values, duplicate IDs, countries, text
lengths, formatting patterns, the ground-truth match distribution, the one-match
assumption check (`ONE_MATCH_ASSUMPTION_each_S2S3_has_at_most_one_S1`), noise statistics
on true pairs vs random non-pairs, and how often true pairs share cheap blocking keys.
It never modifies the TSVs. Progress messages go to stderr and the JSON to stdout.

## 3. Local evaluation (macro F0.5)

```bash
python3 code/business_entity_resolution/src/evaluation.py \
    --predictions path/to/matching_results.tsv \
    --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
    [--candidates path/to/candidate_pairs.tsv] [--only-predicted] [--json]
```

How it scores (identical to the official definition):

| Truth T | Prediction P | Score |
|---|---|---|
| empty | empty | 1.0 |
| empty | non-empty | 0.0 |
| non-empty | empty | 0.0 |
| non-empty | non-empty | F0.5 = 1.25·p·r / (0.25·p + r), with p = \|P∩T\|/\|P\| and r = \|P∩T\|/\|T\| (0 if no overlap) |

The final score is the **macro** mean over all evaluated S1 entities, singletons included.

- Default: every S1 entity in the ground-truth file is evaluated, and one with no prediction row counts
  as empty (with a warning, because the official validator would reject such a file).
- `--only-predicted`: evaluate only the S1 entities present in the predictions, e.g. a held-out fold.
- `--candidates`: adds the blocking diagnostics. These are the pair recall ceiling, the *oracle* macro F0.5
  (the best score a perfect matcher could reach on these candidates), and the candidates per S1
  (mean/median/p95/max). The last one matters because a smaller candidate set is ranked higher.
- The parser tolerates a UTF-8 BOM, CRLF line endings, quotes, spaces and trailing commas, and warns
  about duplicate rows or repeated IDs.

Tests (stdlib `unittest`):

```bash
cd code/business_entity_resolution && python3 -m unittest discover -s tests -v
```

This runs every test. `test_io_dataset.py` loads the full dataset (about 80 s, ~6 GB RAM) and is
skipped when no dataset is found. Its cache check is skipped until the cache below is built.

## 4. Data layer (`src/ber`)

`ber.config.find_data_dir()` looks for the dataset in this order: an explicit `data_dir`, the
`BER_DATA_DIR` environment variable, `student_resource/dataset/`, then
`*_student_resource/student_resource/dataset/` (the official zip extracted in place). The data is
read in place and never copied.

Generated files go to the work directory: an explicit `work_dir`, else `BER_WORK_DIR`, else
`<repo>/work/` (gitignored). It may not overlap the dataset directory.

Build the Parquet cache once (about 1.5 min; it streams block by block, so memory stays low):

```bash
cd code/business_entity_resolution/src
python -m ber.io build-cache            # [--data-dir DIR] [--work-dir DIR] [--force]
```

This writes `work/cache/raw_v1/*.parquet` (zstd, ~1.2 GB) and `manifest.json`. The Parquet files
hold exactly the parsed raw values. A cache entry is used only while the raw file's size and
modification time still match the manifest; otherwise the loaders read the TSV. The cache can be
deleted at any time and rebuilt with the same command. Loading every split and the ground truth
from the cache takes ~15 s at a peak of ~3.5 GB of RAM.

```python
from ber.config import Config
from ber import io

cfg = Config.discover()                      # or Config.discover("path/to/dataset", "path/to/work")
s1 = io.load_source(cfg, "test", 1)          # entity_id, business_name, business_address, country, source
targets = io.load_targets(cfg, "test")       # S2 + S3 in one frame, `source` = 2 or 3
gt = io.load_ground_truth(cfg)               # gt.s1 (id, n_matches), gt.pairs (s1, target, target_source)
for chunk in io.iter_source_chunks(cfg, "train", 2, block_size=64 << 20):
    ...
```

- Text columns are `string[pyarrow]`, `country` is a category and `source` is `int8`.
  Train S1+S2+S3 take about 1.4 GB in memory and load in about 9 s.
- Quotes are read literally, the same way the official validator splits lines. Some names start
  with `"` (132 differ in test S1 from a default `pd.read_csv(sep="\t")`, which strips them).
  Row counts and IDs are the same either way.
- Empty S2/S3 addresses stay `""` (not null). A bad header, a wrong ID prefix or a row with the
  wrong number of fields raises an error.

## 5. Normalization (`src/ber/normalize`)

Build after the raw cache (about 10 min on 8 cores; `--workers` defaults to CPU count - 1):

```bash
cd code/business_entity_resolution/src
python -m ber.normalize.cache build     # [--workers N] [--force] [--data-dir DIR] [--work-dir DIR]
```

This writes `work/cache/norm_v1/{split}_source{n}.parquet` and a manifest. An entry is rebuilt
when its raw TSV or raw-cache file changes, when `config.NORM_CACHE_VERSION` changes, or when
any file of `ber/normalize/` changes (a SHA-256 of the code is stored). Row order, IDs,
countries, sources and both raw strings are copied through unchanged.

| Column | Content |
|---|---|
| `name_raw`, `address_raw` | the raw strings, unchanged |
| `name_norm` | NFKC, casefold, zero-width characters removed, apostrophes dropped (French elisions split), `&` → and/et, dotted initialisms joined, other punctuation → space; accents kept |
| `name_fold` | `name_norm` without Latin diacritics (Indic text untouched) |
| `name_stripped`, `legal_forms` | `name_fold` without the legal forms of `legal_forms.py`, and the canonical forms removed |
| `name_key` | sorted unique tokens of `name_stripped` |
| `name_script`, `address_script` | `empty`, `latin`, one of nine Indic scripts, `other` or `mixed` |
| `address_missing` | nothing real left after dropping empty and `<NULL>`/`NULL` components |
| `address_ws`, `address_punct`, `address_norm` | whitespace-collapsed raw; cleaned and folded; plus canonical numbers and expanded abbreviations |
| `postcode`, `house_numbers`, `unit_numbers`, `address_numbers` | extracted numbers (space-separated) |

Text columns are space-separated, so `.split()` gives the tokens. Indic text is **not**
transliterated here. Load with `ber.normalize.cache.load_normalized(cfg, "train", 2)`.

## 6. Retrieval experiments (`src/ber/tokens.py`, `src/ber/retrieval`, `src/ber/cv.py`)

Targets (S2+S3 of one split) are indexed per country: sorted vocabulary, document frequency,
IDF and an inverted CSR matrix per field (`name_fold`, `name_stripped`, `address_norm`, and
`name_key` as one exact key). S1 records are queries: IDF-weighted CSR rows mapped into their
country's vocabulary. A channel suppresses tokens above its DF cap (queries only; nothing is
deleted), scores `queries @ inverted` in chunks (IDF sum of shared tokens) and optionally keeps
the top-K per query. A strategy is a union of channels with summed scores and a channel mask.

```bash
cd code/business_entity_resolution/src
python ../experiments/e02_retrieval.py          # ~25 min, ~6 GB RAM; [--limit N] for a trial
```

It writes the 5-fold split (`work/cache/cv_v1/train_s1_split.parquet`, dev = 100k S1 of fold 0)
and `work/experiments/retrieval/{strategy_results,candidate_stats,recall_by_stratum,df_summary,run_meta}.json`.
This is the internal retrieval pool only; `candidate_pairs.tsv` is produced in a later phase.

## 7. Stage-1 ranking experiments (`src/ber/ranker.py`, `src/ber/retrieval/indic.py`)

```bash
cd code/business_entity_resolution/src
python ../experiments/e03_indic.py      # ~30 min: Indic skeleton channels vs G on the dev subset
python ../experiments/e03_ranker.py     # ~1 h: train on 100k fold-1..4 S1, compare IDF vs LightGBM on dev
```

The ranker only re-orders retrieved candidates. Features are cheap and vectorized (per-channel
scores/ranks/containment, union score/rank, number agreement, source, country, scripts, missing
flags, token counts, key DF). Configurations are chosen on an S1-level holdout of the training
folds; the fold-0 dev subset is only used for the final comparison. Models and metrics are
written to `work/experiments/ranker/` (gitignored).

## 8. Validate a submission (official script)

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## 9. End-to-end reproduction (data → blocking → matching → output)

Every command runs from `code/business_entity_resolution/src` with the dataset in place
(section 1) and `requirements.txt` installed. Generated files go to `<repo>/work/`
(gitignored); the models are trained by these scripts, not shipped.

```bash
cd code/business_entity_resolution/src

# 1. Data layer and normalization caches (sections 4-5)
python -m ber.io build-cache
python -m ber.normalize.cache build

# 2. Retrieval study; also writes the 5-fold S1 split used by every later step (section 6)
python ../experiments/e02_retrieval.py

# 3. Stage-1 LightGBM ranker on the G+Indic pool -> work/experiments/ranker/model_GI_unweighted.txt
python ../experiments/e03_ranker.py

# 4. Stage-2 LightGBM matcher on the locked stage 1 (400 trees, cap 500, top 50);
#    picks the threshold (0.63) on the fold-0 dev subset
#    -> work/experiments/matcher/full_t60000_v20000/{stage2_model.txt,features.json,config.json}
python ../experiments/e04_matcher.py --train 60000 --valid 20000 --out-name full_t60000_v20000

# 5. Test inference: both submission files for all 1,732,544 test S1, then the official validator
python -m ber.predict --split test --run-name full --out ../../../output --validate
```

Step 5 never loads labels (the ground-truth readers raise inside `ber.predict`). It checks the
stage-1/stage-2 model hashes and feature lists against `config.json`, scores the test S1 in
resumable blocks of 500 (checkpoints in `work/inference/test/full/blocks/`; rerun the same
command after an interruption), then applies threshold 0.63 and the one-target → one-S1 rule
over the whole test set at once. On a 4-core laptop it needs ~8 GB RAM, ~4 min to build the
index and ~10-17 ms per S1 **on AC power** (about 5-8 h for the test set; roughly 4x slower on
battery). `experiments/check_predict_repro.py` checks that `ber.predict` reproduces the
e04 pipeline exactly on training S1.

Optional helpers (not needed for the submitted files):

- `python -m ber.score_blocks --ref-run full --run-name <name> --start A --stop B` scores plan
  blocks `[A, B)` in a second process with identical results (`--compare` checks this against
  the reference run's checkpoints).
- `python -m ber.assemble_partial --run-name full --out <dir> --validate` writes a valid
  submission from the blocks finished so far (unscored S1 get empty rows).
