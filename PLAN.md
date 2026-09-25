# Amazon ML Challenge 2026: Business Entity Resolution

> **Not authoritative.** This is the team's working plan. The verified official rules are in
> `REQUIREMENTS_CHECKLIST.md`; where the two conflict, the checklist and the official documents win.
> The design sections below are provisional until the dataset analysis (`analyze_dataset.py`) has run.

## The task in one paragraph

We get business records (`entity_id`, `business_name`, `business_address`, `country`) from 3 sources.
Source 1 (S1) is deduplicated. For **every S1 record** we must list all S2/S3 records that are the
same real-world business (can be zero, one or many). Train has labels
(`train_ground_truth.tsv`); test does not. Train covers **US + India**; test also has **France**, which
we have never seen in training.

## Scoring: what actually matters

- **Macro F0.5 per S1 entity**, averaged over all S1 entities. Precision counts 2x as much as recall.
- **Singletons** (S1 entities with no true match): an empty prediction scores **1.0**, and even one wrong match scores **0.0**.
- One wrong extra match hurts a lot. Example: 1 correct match out of 2 true matches, no wrong ones, gives F0.5 = 0.83. The same plus one wrong match gives F0.5 = 0.50.
- So we **only output matches we are confident in**, and "no match" must be a first-class prediction.

## Hard rules

| Rule | Consequence |
|---|---|
| Output files are TSV (`sep="\t"`), ID lists comma-separated, no quotes | Otherwise the file is rejected |
| Exactly one row per test S1 entity (including France), empty list allowed | Otherwise the file is rejected |
| Only existing S2/S3 test IDs, no duplicates in a list, no duplicate S1 rows | Otherwise the file is rejected |
| Final model: **MIT or Apache-2.0 licence, ≤ 8B parameters** | Otherwise we are disqualified |
| **No external data**: no business lookups, no geocoding APIs, no internet augmentation | Otherwise we are disqualified |
| Max **5 leaderboard submissions per day** (15 total) | Every submission must be validated locally first |
| Final rank = **private leaderboard** | Trust local CV over the public leaderboard |

Always run `python3 utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test`
(from `student_resource/`) before uploading.

## Deliverables

1. `matching_results.tsv`: uploaded to the leaderboard portal.
2. `<team_name>_submission.zip`, containing:
   ```
   output/matching_results.tsv
   output/candidate_pairs.tsv          # final candidate set fed to the model (superset of matches)
   code/business_entity_resolution/src/
   code/business_entity_resolution/README.md          # exact end-to-end reproduction steps
   code/business_entity_resolution/requirements.txt   # pinned versions
   Documentation_template.md           # filled-in methodology write-up
   ```
3. The methodology doc must cover the methodology, the blocking strategy, the model and features, and anything else relevant.
   Note: the problem statement says "no page limit", but the guidelines say "1–2 pages". **Ask via the query form.**

## Pipeline design

```
raw TSVs
  -> 1. normalise   (lowercase, unicode/accents fold, punctuation, &->and,
                     legal-suffix + street-type abbreviation maps incl. French: rue, av, bd, sarl, sas)
  -> 2. block       (union of several cheap retrievers per S1 record, top-K each)
  -> 3. pair features
  -> 4. pair classifier  (GBDT; optional cross-encoder)
  -> 5. decide      (thresholds + optional 1-to-1 constraint IF the data supports it, tuned for macro F0.5)
  -> output/candidate_pairs.tsv, output/matching_results.tsv
```

### 2. Blocking (this sets our recall ceiling)
- TF-IDF **char 3–5-gram** on name, and on name+address, then nearest neighbours (top-K per S1 record).
- Multilingual **sentence embeddings** (Apache/MIT model, e.g. `intfloat/multilingual-e5-small/base`, LaBSE)
  on name+address, then kNN.
- Token blocking on rare tokens and numbers (postcode/PIN, house numbers).
- Restrict to the same country where the country label is reliable (check the data first; don't hard-code US/India).
- Track **pair recall** and **candidates per S1** on validation (`evaluation.py --candidates`).
  **Official update:** a smaller candidate set per S1 is ranked higher, and `candidate_pairs.tsv` is reviewed.
  So choose K from the measured recall vs set-size curve, not a fixed target.
- **Scale:** the validator says the full test set is **~1.7M entities**. Blocking has to be sub-quadratic:
  partition by country (and a coarse geo key such as postcode prefix or city token), use sparse top-K
  (`sparse_dot_topn`) or ANN (FAISS/hnswlib), keep K as small as the recall curve allows (the old ≈10–30 guess is superseded), and cache everything to disk.
  Embedding 1.7M strings on CPU is slow (probably hours), so prefer a small model, or a free GPU (Colab/Kaggle) for that step.
- Write outputs with plain Python string joins, not pandas quoting, and validate with `--check-ids` only when memory allows.

### 3. Features (per S1–candidate pair)
- Name: Jaro-Winkler, Levenshtein ratio, token-set/sort ratio, Jaccard, TF-IDF cosine, and the same after removing legal suffixes;
  acronym match; first-token match.
- Address: the same similarities, plus postcode/PIN equal or missing, numeric-token overlap, city token overlap.
- Embedding cosine (name, address, combined).
- Rank features: the candidate's rank among this S1's candidates, and the gap to the best score; the S1's rank among the candidate's reverse neighbours.
- Source (S2 vs S3), missing-field flags. Country only as a generic "same country" flag, never one-hot.

### 4. Model
- LightGBM/XGBoost binary classifier on pairs (fast on CPU, strong baseline).
- Stretch goal: fine-tune a small multilingual cross-encoder (≤ 8B, Apache/MIT) on top pairs, then blend its score into the GBDT.

### 5. Decision layer (tuned directly on macro F0.5)
- A global threshold `t` on p(match), probably high because F0.5 favours precision.
- The **singleton gate**: if max p over an S1's candidates is below `t_empty`, predict empty.
- Keep extra matches only if p > t and p is within a margin of the best one.
- **Hypothesis, not a rule:** each S2/S3 record belongs to at most one S1 (S1 is deduplicated). Only if the
  ground truth confirms it (`ONE_MATCH_ASSUMPTION_…` in `analyze_dataset.py`), assign each S2/S3 to its best-scoring S1.
- Grid-search `t`, `t_empty` and the margin on out-of-fold predictions.

### Validation
- Split **S1 entities** into 5 folds (GroupKFold). S2/S3 records stay in the candidate pool.
- A local scorer that exactly replicates macro F0.5 (singletons included).
- France generalisation: hold out one country (train on US, test on India) as a proxy check for unseen-country robustness.

## Timeline (IST). Window: Fri 25 Sep 00:00 to Sun 27 Sep 23:59

| When | Goal | Submissions |
|---|---|---|
| **Day 1 (Fri) evening** | Repo, data loading, EDA, scorer, validator, normalisation, TF-IDF blocking, simple-rules baseline, **first valid submission** | 1–2 |
| **Day 2 (Sat)** | Embedding blocking, full feature set, LightGBM, OOF threshold tuning, 1-to-1 assignment (if validated) | 3–5 |
| **Day 3 (Sun) until ~18:00** | Improvements (cross-encoder / more features), France robustness checks, final submission | 3–5 |
| **Day 3 (Sun) 18:00–23:00** | Freeze code, clean README + requirements, fill in Documentation_template.md, build the zip, dry-run reproduction | – |

Keep a submission log (date, commit hash, local CV, public LB) in `experiments.md`.

## Team split (suggested)
- **Person A**: blocking + recall measurement.
- **Person B**: features + LightGBM + threshold tuning.
- **Person C**: normalisation dictionaries (US/India/France), EDA, documentation + final zip.

## Questions to send via the query form
1. Doc length: "no page limit" (problem statement) vs "1–2 pages" (guidelines)?
2. Are pretrained open models (e.g. multilingual sentence embeddings under Apache/MIT) allowed for features and blocking?
3. Are hand-written abbreviation/normalisation dictionaries (street types, legal suffixes) acceptable? They aren't external *data lookup*.
