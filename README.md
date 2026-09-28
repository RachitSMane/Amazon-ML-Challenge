# Amazon ML Challenge 2026: Business Entity Resolution

**Team CloundForge**: a label-free, CPU-only entity-resolution pipeline built on sparse retrieval
and gradient-boosted trees.

## Overview

Business records arrive from three independent sources that share no identifiers. This project decides
which records refer to the same real-world business. **Source 1 (S1)** is the deduplicated reference
source; for every S1 record the pipeline returns the matching records from **Source 2 (S2)** and
**Source 3 (S3)**.

## Challenge objective

- **Input:** S1 reference records, plus S2/S3 target records (`entity_id`, `business_name`, `business_address`, `country`).
- **Output:** for each S1 record, every S2/S3 record that is the same business. There can be **zero, one or many**.
- **Difficulties:**
  - Names have legal-form variants (*Pvt Ltd* / *Private Limited*, *SARL*), `&`/*and*, punctuation, accents and typos.
  - Addresses have abbreviations, `<NULL>` components, missing postcodes and zero-padded numbers.
  - Some target names are written in **Indic scripts**.
  - The test set contains a country (**France**) that never appears in training.
- **Metric:** macro-averaged **F0.5** per S1 record. An S1 record with no true match scores 1.0 only when nothing is predicted. Precision counts twice as much as recall.

The full official statement is in [`student_resource/README.md`](student_resource/README.md).

## Solution overview

```
raw TSVs ─► normalization ─► G+Indic sparse retrieval ─► Stage 1: LightGBM ranker ─► Stage 2: LightGBM matcher ─► decision rule
            (names, addresses,   (per-country inverted     (union-rank cap, top-K      (62 pair features,          (threshold + each target
             legal forms)         indexes, IDF scoring)      per S1 = candidate set)     p(match))                   goes to at most one S1)
                                                                   │                                                      │
                                                                   └──► candidate_pairs.tsv                               └──► matching_results.tsv
```

| Stage | Module | What it does |
|---|---|---|
| Normalization | `ber.normalize` | Deterministic cleaning of names and addresses; cached as Parquet |
| Candidate retrieval | `ber.tokens`, `ber.retrieval` | IDF-weighted sparse retrieval over several channels, indexed per country |
| Stage 1: ranking | `ber.ranker` | LightGBM re-ranks the retrieved candidates; the top-K become the candidate set |
| Stage 2: matching | `ber.matcher` | LightGBM binary classifier over pair features gives p(match) |
| Prediction | `ber.predict` | Resumable block-wise inference; writes both submission files |

### Candidate retrieval (G + Indic)

Targets (S2 + S3) are indexed **per country** as inverted CSR matrices, and each S1 record queries only
its own country's pool. A channel scores a target as the sum of the IDF of the tokens they share. Tokens
above the channel's document-frequency cap are ignored for that channel. The **G+Indic** union used in
`ber.predict` combines four channels:

| Channel | Field | Purpose |
|---|---|---|
| C1 `name` | `name_fold` (normalized, accent-folded name), DF cap 10,000 | Name token overlap |
| C2 `addr` | `address_norm`, DF cap 1,000 | Address token overlap |
| K `key` | `name_key` (legal forms stripped, sorted tokens) | Exact "strong name" match |
| Indic `ikey` | phonetic skeleton of the stripped name | Latin S1 names vs Indic-script target names |

Channel scores are merged per (S1, target) with a channel mask and ranked by union score.

### Stage 1: LightGBM ranking

`ber.ranker` builds cheap, vectorized pair features:
- per-channel scores, ranks and containment,
- the union score and rank,
- number agreement,
- source, scripts, missing flags, token counts and key document frequencies.

A LightGBM model (`model_GI_unweighted.txt`, first 400 trees) scores the candidates that fall within the
union-rank **cap**, and keeps the **top-K** per S1. That top-K list is exactly what Stage 2 scores and
what `candidate_pairs.tsv` contains. The locked configuration in `ber/matcher.py` is **cap 500 / top 50**.
A later experiment used **cap 1000 / top 200** (see [Results](#results--experiments)).

### Stage 2: matching

`ber.matcher` computes **62 features** for each candidate pair:
- **Stage-1 ranking context:** the Stage-1 score and rank, the gap to the best candidate, and related values.
- **Stage-1 pair features:** everything except `country`, which is dropped so the model works for the unseen test country.
- **RapidFuzz similarities** on the normalized name, the legal-form-stripped name, the address and the number tokens.

A LightGBM binary classifier outputs p(match). The **decision rule** keeps a pair when p ≥ threshold **and**
the pair is the highest-scoring S1 for that target. Each S2/S3 record goes to at most one S1, a property
the training ground truth satisfies. The threshold is tuned for macro F0.5 on held-out development data.

## Data processing

`ber.normalize` is deterministic, uses explicit rule lists, and makes no geographic or external lookups.

- **Text (`text.py`):**
  - NFKC and casefold; zero-width characters and missing-value placeholders removed.
  - Apostrophes handled by country (French elisions split).
  - `&` becomes *and*, or *et* for France; dotted initialisms are joined (`p.v.t.` → `pvt`).
  - Other punctuation becomes a space.
  - Latin diacritics are folded; Indic combining marks are left untouched.
  - Script detection covers Latin, nine Indic scripts, mixed and other.
- **Legal forms (`legal_forms.py`):**
  - Explicit per-country lists: global/US, India, France and Indic-script forms.
  - Suffixes are removed only at the end of a name and prefixes only at the start; a name is never emptied.
- **Addresses (`address.py`):**
  - `<NULL>` components are removed individually and the rest of the address is kept.
  - Numbers are canonicalized (`06252` → `6252`) and abbreviations expanded (`rd` → `road`).
  - Postcodes, house numbers and unit numbers are extracted.
- **Tokens (`tokens.py`):** per-pool and per-country vocabularies, document frequency and IDF, and inverted indexes.
- **Country:** treated as an open set of labels. It partitions the retrieval index but is never one-hot encoded or used as a model feature.

### Indic / multilingual handling

Some true pairs have a Latin S1 name and an Indic-script target name that share no token. Most are English
words written in an Indic script, such as *राम मार्केटिंग* for "Ram Marketing". `ber/retrieval/indic.py`
handles these in two steps:
1. It romanizes the nine Brahmic scripts through one shared table.
2. It reduces both sides to a coarse consonant-class **phonetic skeleton key**.

Matching on this exact key adds the Indic channel to retrieval. It uses no external transliteration
library, and the normalized columns stay unchanged.

## Project structure

```
.
├── README.md                          # this file
├── PLAN.md, REQUIREMENTS_CHECKLIST.md # planning notes and the verified official-rules checklist
├── Documentation_template.md          # methodology write-up (first submission)
├── student_resource/                  # official challenge material
│   ├── README.md                      # official problem statement
│   ├── Documentation_template.md      # blank official template
│   └── utils/validate_submission.py   # official submission validator
├── code/business_entity_resolution/  # the solution (reproducible package)
│   ├── README.md                      # detailed setup and exact run commands (sections 1-11)
│   ├── requirements.txt               # pinned, permissively licensed dependencies
│   ├── src/
│   │   ├── evaluation.py              # local macro F0.5 scorer + candidate diagnostics (stdlib)
│   │   ├── analyze_dataset.py         # read-only dataset analysis (stdlib)
│   │   └── ber/                       # config, io, normalize/, tokens, retrieval/, cv, ranker,
│   │                                  # matcher, predict, score_blocks, rescore, assemble_*
│   ├── experiments/                   # e02 retrieval, e03 Indic + ranker, e04 matcher,
│   │                                  # e05 Stage-2 variants, e06 paired bootstrap, repro check
│   └── tests/                         # unittest suite (11 modules)
├── CloundForge_submission/            # first-submission package: methodology doc + code snapshot
├── submission_v2/                     # v2 methodology doc + hashes of the v1 files
└── submission_hybrid/                 # v3/v1 hybrid methodology doc
```

Caches, models and checkpoints go to `work/` (git-ignored). The submission TSVs and zip archives are
excluded by the `*.tsv` / `*.zip` rules in `.gitignore`.

## Installation

Python ≥ 3.10 is required. The dependencies are pinned and permissively licensed (MIT / Apache-2.0 / BSD-3):
numpy, pandas, pyarrow, LightGBM, RapidFuzz and SciPy.

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
```

Place the official dataset at `student_resource/dataset/{train,test}/` (see
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md), section 1).

## How to run

All pipeline commands run from `code/business_entity_resolution/src`, with the dataset in place:

```bash
cd code/business_entity_resolution/src

# 1. Raw and normalized Parquet caches
python -m ber.io build-cache
python -m ber.normalize.cache build

# 2. Retrieval study; also writes the 5-fold S1 split used later
python ../experiments/e02_retrieval.py

# 3. Stage-1 LightGBM ranker on the G+Indic pool
python ../experiments/e03_ranker.py

# 4. Stage-2 LightGBM matcher (trains the model and picks the threshold on dev data)
python ../experiments/e04_matcher.py --train 60000 --valid 20000 --out-name full_t60000_v20000

# 5. Test inference: writes candidate_pairs.tsv + matching_results.tsv and runs the official validator
python -m ber.predict --split test --run-name full --out ../../../output --validate
```

Inference is resumable: rerun the same command after an interruption. The commands for the v2 and hybrid
submissions, and helpers such as `ber.score_blocks`, `ber.rescore` and `ber.assemble_hybrid`, are in
sections 10–11 of the [code README](code/business_entity_resolution/README.md).

**Local evaluation** against the training ground truth:

```bash
python code/business_entity_resolution/src/evaluation.py \
    --predictions path/to/matching_results.tsv \
    --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
    [--candidates path/to/candidate_pairs.tsv] [--only-predicted]
```

**Official validator:**

```bash
cd student_resource
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Testing

The suite uses the standard-library `unittest` framework and covers evaluation, I/O, normalization,
tokens, sparse top-K, retrieval, CV folds, the ranker and the matcher. Install `requirements.txt` first.

```bash
cd code/business_entity_resolution
python -m unittest discover -s tests -v
```

- `test_evaluation.py` needs only the standard library.
- `test_io_dataset.py` and `test_normalize_dataset.py` run against the real dataset and its caches. They are skipped when those are absent.

## Submission

Each submission consists of two tab-separated files for all test S1 records:

| File | Content |
|---|---|
| `matching_results.tsv` | `source1_entity_id` → final matched S2/S3 IDs (leaderboard file) |
| `candidate_pairs.tsv` | `source1_entity_id` → the exact top-K list scored by Stage 2 (matches are always a subset) |

- **Final package:** `CloundForge_submission/` follows the official structure. It holds the filled-in methodology document and a copy of `code/business_entity_resolution/` as used for the first submission.
- **Outputs are not in git:** the generated TSVs and zip archives are git-ignored, so they are not part of this repository. `submission_v2/v1_hashes_before_v2.txt` records the SHA-256 of the first submission's files.
- **Which methodology document goes with which submission:**
  - the root `Documentation_template.md` and the one in `CloundForge_submission/`: v1
  - `submission_v2/`: v2
  - `submission_hybrid/`: v3/v1 hybrid

## Results / experiments

The figures below are **internal validation results** reported in this repository's methodology documents
and code README. They were measured on a fixed held-out development subset of 20,000 training S1 records
(fold 0), using the local macro-F0.5 evaluator. **They are not official leaderboard scores.**

| Configuration | Stage-1 cap / top-K | Stage-2 training S1 | Threshold | Dev macro F0.5 |
|---|---|---|---|---|
| v1 | 500 / 50 | 60k | 0.63 | 0.92694 |
| v2 (Stage 2 retrained) | 500 / 50 | 100k | 0.69 | 0.92813 |
| v3 | 1000 / 200 | 60k | 0.66 | 0.93766 |

- **Candidate recall:**
  - v1's top-50 lists contain 0.890 of the true dev pairs (oracle F0.5 0.953).
  - v3's top-200 lists contain 0.920 (oracle 0.968).
- **v1 candidate set:** the v1 `candidate_pairs.tsv` is documented as 85,166,352 pairs for 1,732,544 test S1, about 49 per S1.
- **Hybrid submission:** v3 is too slow for a full test pass (~20 ms per S1). The second submission therefore combined v3 on the test blocks scored before the deadline with v1 everywhere else (`ber.assemble_hybrid`).
- **v2 gain:** a paired bootstrap put v2's improvement over v1 at +0.00119 (95% CI [+0.00037, +0.00197]).

## Reproducibility

- **Deterministic:** normalization, retrieval, S1-level cross-validation folds (stratified by country × number of matches) and LightGBM training are all deterministic. Every selection was made on training folds, never on the leaderboard.
- **Label-free inference:** `ber.predict` disables the ground-truth loaders. It also checks the model hashes and feature lists against the saved `config.json` before scoring.
- **Reproduction check:** `experiments/check_predict_repro.py` verifies that `ber.predict` reproduces the experiment pipeline exactly on training S1.
- **Models are not shipped:** they are retrained by the commands above. Step-by-step instructions, runtimes and memory needs are in the [code README](code/business_entity_resolution/README.md).
- **Challenge rules:** no external data, APIs or pretrained models are used. The only model family is LightGBM (MIT, far below the 8B-parameter limit).

## Important note

**The raw challenge dataset is not included in this repository.** It is restricted challenge material
and very large: about 1.73 M test S1 records against about 9.97 M S2+S3 targets. `.gitignore` excludes
`student_resource/dataset/`, every `*.tsv` / `*.zip`, and the `work/` caches and models. To
reproduce the results, obtain the official `student_resource` archive and place its `dataset/` folder
as described above.

## Team

**CloundForge**

- Rachit Mane
- Prithvi.W
- Pranav .V.Mennon
- Akash.R
