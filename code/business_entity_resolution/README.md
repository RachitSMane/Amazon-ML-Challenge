# business_entity_resolution

The team's solution for the Amazon ML Challenge 2026 (Business Entity Resolution).
This folder is what ships in the final zip as `code/business_entity_resolution/`
(see `student_resource/README.md`, "Final Submission Package").

> **Status: preparation phase.** Only the data-understanding and evaluation tools exist.
> The blocking/matching pipeline is not implemented yet.

## Layout

```
code/business_entity_resolution/
├── README.md             # this file: setup + exact run instructions
├── requirements.txt      # pinned dependencies (currently none: stdlib only)
├── src/
│   ├── evaluation.py     # local macro F0.5 scorer + candidate-set diagnostics
│   └── analyze_dataset.py# read-only aggregate dataset analysis (EDA)
└── tests/
    └── test_evaluation.py
```

All commands below run from the **repository root**, with Python ≥ 3.8.

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

## 4. Validate a submission (official script)

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
