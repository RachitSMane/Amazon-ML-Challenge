# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CloundForge  
**Team Members:** Rachit Mane, Prithvi.W, Pranav .V.Mennon, Akash.R  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A three-stage pipeline: IDF-weighted sparse retrieval over normalized names, addresses and two exact name keys (including a phonetic skeleton key for Indic scripts) generates candidates. A LightGBM ranker keeps the top 50 per Source-1 record, and a second LightGBM classifier with 62 string, number and ranking features decides the matches at a precision-oriented threshold. A global rule assigns each Source-2/3 record to at most one Source-1 record. On a held-out development set of 20,000 training S1 the pipeline scores **macro F0.5 = 0.927** (pair precision 0.985, pair recall 0.855).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Precision-heavy, singleton-aware metric.** F0.5 is macro-averaged per S1, and an S1 with no true match scores 1.0 only if nothing is predicted. Wrong merges are therefore more costly than missed links, and predicting "empty" has to be a real option for every record.
- **Name noise:** legal-form variants (Pvt Ltd / Private Limited, Inc / Corporation, SARL / SAS), `&` vs and/et, punctuation and dotted initialisms, French elisions (`d'alzon`), accents, word order, typos, and Indic-script names with transliteration and zero-width-joiner differences.
- **Address noise:** `<NULL>` placeholders appear as individual comma-separated components rather than whole values, abbreviations (Rd/Road, St/Street) vary, house numbers are zero-padded differently across sources (`06252` vs `6252`), and PIN/ZIP codes or whole components are missing.
- **Open country set:** the test set adds France, which is absent from training. Every component is country-agnostic, and country only partitions the target index. Country-specific rules are limited to explicit legal-form and elision lists.
- **Scale:** 1.73 M test S1 against 9.97 M S2+S3 targets. This rules out pairwise comparison and requires vectorized sparse retrieval with a bounded candidate set per S1.

### 2.2 Solution Strategy
**Approach Type:** Blocking (sparse IDF retrieval) + learned ranker + learned classifier + global assignment constraint  
**Core Innovation:**
- A union of several capped IDF retrieval channels plus an Indic phonetic-skeleton key.
- A two-stage LightGBM ranker and matcher, where the stage-1 scores and ranks become stage-2 features.
- An exact, deterministic, resumable block-wise inference that never touches labels.

All model and threshold selection uses deterministic S1-level cross-validation. The folds are stratified by country × number of true matches. Fold 0 contains the development subset and is used only for final evaluation and threshold choice.

---

## 3. Candidate Generation (Blocking)
Normalization (`ber.normalize`) is deterministic: NFKC, casefold, invisible-character removal, and placeholder removal at the component level. Legal forms are stripped using explicit per-country lists. Addresses get number canonicalization and abbreviation expansion. Targets are indexed per country as inverted CSR matrices, and each channel scores a query as the IDF sum of the tokens it shares with a target. Tokens above a channel's document-frequency cap are suppressed for that channel.

- **Blocking keys used ("G+Indic" union):**
  - C1: name tokens, DF cap 10,000.
  - C2: address tokens, DF cap 1,000.
  - K: exact strong-name key (legal forms stripped, tokens sorted).
  - Indic key: exact sorted phonetic-skeleton key of the stripped name, matched against Indic or mixed-script targets.

  Channel scores and a channel mask are merged per (S1, target), and candidates are ordered by union IDF score.
- **Candidate pairs generated:**
  - Retrieval returns about 4,000 targets per S1 on average.
  - Stage 1 re-ranks the best 500 by union rank and keeps the **top 50** per S1.
  - `candidate_pairs.tsv` holds exactly these top-50 lists: **85,166,352 pairs** for 1,732,544 test S1 (49.2 per S1 on average). Only 677 S1 have no candidate.
- **How you ensured true matches were not lost:**
  - Every channel and union was measured on a 100k-S1 development subset of the training data: pair recall, S1 recall, oracle F0.5, and recall by country, script, target source and missing address.
  - The G union reaches pair recall 0.931. The Indic key raises this to 0.953, with an oracle F0.5 of 0.981.
  - The stage-1 ranker (below) keeps far more true pairs in the top 50 than IDF order alone does: pair recall 0.892 vs 0.827 on the training-fold holdout.

---

## 4. Matching Model

**Features used (62, identical order in training and inference):**
- Name features: per-channel IDF score, rank, query/target token-coverage fractions and weighted Jaccard (name, strong key, Indic key); RapidFuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the normalized name and on the legal-form-stripped name; equality of legal form; equality of the first stripped token; token counts; the query's maximum token IDF; key document frequencies.
- Address features: per-channel IDF score, rank, coverage fractions and weighted Jaccard on the address; RapidFuzz ratio, token-set and partial ratio; token-set similarity over numbers only; agreement and conflict flags for house number, unit number and postcode; missing-address flags.
- Other: union score, rank and relative score; number of channels hitting the pair; log candidate count; target source (S2/S3); query and target scripts; and the stage-1 raw score, rank, gap to the best, best score, gap between the 1st and 2nd scores, and number of near-top candidates.

**Model type:** Two LightGBM gradient-boosted tree models (MIT license, well under 8 B parameters; no neural or pretrained model).
- **Stage 1 (ranker):** binary LightGBM on cheap vectorized pair features. It was trained on 100k S1 from folds 1–4 using all positives and sampled negatives. The unweighted configuration, the first 400 trees and a cap of 500 were chosen on a holdout from the training folds.
- **Stage 2 (matcher):** binary LightGBM (learning rate 0.05, 63 leaves, min 200 rows per leaf, deterministic). It was trained on every top-50 candidate of 60,000 further fold-1–4 S1, excluding the stage-1 training S1 so that the stage-1 scores are out-of-sample. That is 2.65 M rows, 6.3% of them positive. Early stopping at 1,336 trees gave a holdout AUC of 0.9997.

**Threshold selection method:** macro-F0.5 optimization over a threshold grid on the 20,000-S1 fold-0 development subset gave **p ≥ 0.63**. After thresholding, each target goes only to the S1 that gives it the highest probability (the one-target → one-S1 rule), applied globally over the whole test set. An S1 with no pair above the threshold gets an empty list.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.92694** on the 20,000-S1 fold-0 development subset (threshold 0.63; 0.9247 at 0.5 and 0.9254 at 0.8, so the choice is not fragile). Pair precision is 0.9847 and pair recall 0.8545. An oracle that is perfect on the top-50 candidates would score 0.9534.
- **Where errors come from:**
  - The top-50 lists contain 89.0% of all true pairs, so about 11% of true pairs are lost in retrieval and ranking.
  - The matcher recovers 85.5% of all true pairs, so another ~3.5% are lost to the threshold.
  - The gap to the oracle is therefore mostly recall. False merges are rare by design (precision 0.985).
- **Common false positives (wrong merges):** we did not do a manual audit of individual errors in the time available. By design, the riskiest pairs are near-identical names whose address evidence is missing or ambiguous (chain outlets, common names), because name similarity alone cannot separate them. The house/unit/postcode conflict features and the one-target → one-S1 rule (which removed 93,821 above-threshold pairs on test) target exactly this case.
- **Common false negatives (missed matches):** quantitatively, most missed pairs never reach the top 50 (about 11% of true pairs, versus about 3.5% lost at the threshold). The expected causes are name changes that no retrieval channel shares a token or key with (DBA/trade names, heavy abbreviation, transliterations beyond the skeleton key), made worse when the address is missing.
- **Test-set output:**
  - 5,316,134 predicted pairs: 2,607,791 in S2 and 2,708,343 in S3.
  - 1,608,147 S1 with at least one match, and 124,397 S1 predicted empty (7.2%, matching the 7.6% predicted empty on dev).
  - The official validator reported PASS.

---

## 6. Conclusion
A precision-first cascade reached macro F0.5 ≈ 0.927 on held-out data with only gradient-boosted trees: normalization, wide multi-channel sparse retrieval, a learned top-50 ranker, and a calibrated matcher with a global one-to-one target constraint. The largest remaining headroom is candidate recall (oracle 0.953). We also learned that exact reproducibility (hash-checked models, a label-free inference path, deterministic blocks) and operational robustness (resumable checkpoints; not running a 7-hour job on battery) matter as much as the model for delivering the full test set on time.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains:
- `src/ber/`, with these submodules:
  - `io`: raw Parquet cache.
  - `normalize`: text, legal forms, address, normalized cache.
  - `tokens` and `retrieval`: per-country inverted indexes, channels, the union and the Indic key.
  - `cv`: stratified S1 folds.
  - `ranker`: stage-1 features and training utilities.
  - `matcher`: stage-2 features and the decision rule.
  - `predict`: end-to-end inference.
- `experiments/e02–e04`: retrieval study, stage-1 ranker, stage-2 matcher.
- `tests/`.
- `README.md` (section 9 gives the exact end-to-end commands) and `requirements.txt` (pinned, permissive licenses only).

Both output files are produced by:
```
cd code/business_entity_resolution/src
python -m ber.predict --split test --run-name full --out ../../../output --validate
```
It runs after the caches, the stage-1 ranker (`e03_ranker.py`) and the stage-2 matcher (`e04_matcher.py --train 60000 --valid 20000 --out-name full_t60000_v20000`) have been built.

### B. Additional Results
| Stage (development data) | Metric |
|---|---|
| Retrieval G (100k dev S1) | pair recall 0.931, oracle F0.5 0.971, ~3,960 candidates/S1 |
| Retrieval G+Indic (2k dev S1) | pair recall 0.953, oracle F0.5 0.981 |
| Top-50 by IDF order (holdout) | pair recall 0.827, oracle F0.5 0.917 |
| Top-50 by stage-1 LightGBM, 400 trees, cap 500 (holdout) | pair recall 0.892, oracle F0.5 0.955 |
| Stage 2 at t = 0.63 (20k dev S1) | **macro F0.5 0.927**, precision 0.985, recall 0.855 |
| Same, without the one-target → one-S1 rule | macro F0.5 0.927 (the rule matters mostly at test scale) |
| Test inference | 1,732,544 S1; 85.2 M candidate pairs; 5.32 M matches; validator PASS |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
