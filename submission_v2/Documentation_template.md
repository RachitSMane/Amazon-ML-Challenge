# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CloundForge  
**Team Members:** Rachit Mane, Prithvi.W, Pranav .V.Mennon, Akash.R  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A three-stage pipeline: IDF-weighted sparse retrieval over normalized names, addresses and two exact name keys (including a phonetic skeleton key for Indic scripts) generates candidates. A LightGBM ranker keeps the top 50 per Source-1 record, and a second LightGBM classifier with 62 string, number and ranking features decides the matches at a precision-oriented threshold. A global rule assigns each Source-2/3 record to at most one Source-1 record. **This is submission v2.** It keeps retrieval, Stage 1 and the candidate set of v1 unchanged, and retrains Stage 2 on 100,000 instead of 60,000 training S1 (threshold 0.69 instead of 0.63). On the same held-out 20,000 development S1, macro F0.5 goes from 0.92694 (v1) to **0.92813** (v2). A paired bootstrap puts the gain at +0.00119, with a 95% CI of [+0.00037, +0.00197].

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

All model and threshold selection uses deterministic S1-level cross-validation. The folds are stratified by country × number of true matches. Fold 0 contains the development subset and is used only for final evaluation and threshold choice. The leaderboard was not used to select anything.

### 2.3 What changed from v1, and why (evidence-driven)
Before changing anything, we measured where v1 loses score on the held-out dev set:

| Where the score is lost (20k dev S1) | Macro F0.5 |
|---|---|
| v1 as submitted (threshold 0.63) | 0.92694 |
| Oracle on the same top-50 candidates (a perfect Stage 2) | 0.95339 |
| Oracle on all retrieved candidates (Stage-1 holdout, 2k S1) | ~0.981 |

- **Threshold:** v1's threshold was already the dev optimum, and the curve is flat (0.9218 at 0.40, 0.9269 at 0.63, 0.9234 at 0.85). Re-tuning it could not help.
- **Larger top-K:** raising top-K from 50 to 100 or 200 lifts the oracle only to about 0.962 / 0.964 (Stage-1 holdout), for 2–4× more candidates. We rejected this because it gives little and makes the candidate set, which is judged separately, much larger.
- **Stage-1 cap and trees:** a larger cap with the full Stage-1 model (cap 2000, 1,662 trees) raises top-50 recall from 0.892 to 0.910 at the same candidate count. It is the most promising retrieval lever, but it costs about 16× the Stage-1 compute and could not be run on the full test set in the time available. We did **not** use it.
- **Stage 2 (chosen):** Stage 2 had been trained on 60k S1. We retrained it on 100k S1 with the same features, parameters and early-stopping protocol, and compared the two on the identical dev S1 (re-scoring v1's model on the new dev rows gives exactly 0.92694 again). v2 is better at every threshold from 0.40 to 0.85, and the paired bootstrap interval excludes zero.

---

## 3. Candidate Generation (Blocking)
Unchanged from v1. Normalization (`ber.normalize`) is deterministic: NFKC, casefold, invisible-character removal, and placeholder removal at the component level. Legal forms are stripped using explicit per-country lists. Addresses get number canonicalization and abbreviation expansion. Targets are indexed per country as inverted CSR matrices, and each channel scores a query as the IDF sum of the tokens it shares with a target. Tokens above a channel's document-frequency cap are suppressed for that channel.

- **Blocking keys used ("G+Indic" union):**
  - C1: name tokens, DF cap 10,000.
  - C2: address tokens, DF cap 1,000.
  - K: exact strong-name key (legal forms stripped, tokens sorted).
  - Indic key: exact sorted phonetic-skeleton key of the stripped name, matched against Indic or mixed-script targets.

  Channel scores and a channel mask are merged per (S1, target), and candidates are ordered by union IDF score.
- **Candidate pairs generated:**
  - Retrieval returns about 4,000 targets per S1 on average.
  - Stage 1 re-ranks the best 500 by union rank and keeps the **top 50** per S1.
  - `candidate_pairs.tsv` holds exactly these top-50 lists, the set Stage 2 runs inference over: **@@CAND_PAIRS@@ pairs** for 1,732,544 test S1, at most 50 per S1. It is identical to v1's candidate set.
- **How you ensured true matches were not lost:**
  - Every channel and union was measured on a 100k-S1 development subset of the training data (pair recall, S1 recall, oracle F0.5, recall by stratum).
  - The G union reaches pair recall 0.931, and 0.953 with the Indic key (oracle F0.5 0.981).
  - Stage 1 keeps 89.0% of all true pairs in the top 50 on the 20k dev S1. On the Stage-1 holdout (2k S1), its top-50 pair recall is 0.892 vs 0.827 for IDF order alone.

---

## 4. Matching Model

**Features used (62, same as v1 and identical order in training and inference):**
- Name features: per-channel IDF score, rank, query/target token-coverage fractions and weighted Jaccard (name, strong key, Indic key); RapidFuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the normalized name and on the legal-form-stripped name; equality of legal form; equality of the first stripped token; token counts; the query's maximum token IDF; key document frequencies.
- Address features: per-channel IDF score, rank, coverage fractions and weighted Jaccard on the address; RapidFuzz ratio, token-set and partial ratio; token-set similarity over numbers only; agreement and conflict flags for house number, unit number and postcode; missing-address flags.
- Other: union score, rank and relative score; number of channels hitting the pair; log candidate count; target source (S2/S3); query and target scripts; and the stage-1 raw score, rank, gap to the best, best score, gap between the 1st and 2nd scores, and number of near-top candidates.
- **Importance (gain, v2 model):** Stage-1 score 0.71, address token-set similarity 0.07, Stage-1 rank 0.04, address number token-set 0.04, then name partial ratio, stripped-name ratio and name token-sort (about 0.016 each). 17 features have near-zero gain. No new features were added in v2, because none could be tested properly and re-run on the full test set in the time available.

**Model type:** Two LightGBM gradient-boosted tree models (MIT license, well under 8 B parameters; no neural or pretrained model).
- **Stage 1 (ranker, unchanged):** `model_GI_unweighted.txt`, trained on 100k S1 from folds 1–4 using all positives and sampled negatives. It uses its first 400 trees, a cap of 500 and the top 50 per S1.
- **Stage 2 (v2):** binary LightGBM (learning rate 0.05, 63 leaves, min 200 rows per leaf, feature fraction 0.9, bagging 0.8, L2 1.0, deterministic; same parameters as v1). It was trained on every top-50 candidate of **100,000** fold-1–4 S1, excluding the Stage-1 training S1 so that the Stage-1 scores are out-of-sample.
  - 4.42 M training rows (6.3% positive).
  - A 10% S1-hash early-stopping holdout (0.49 M rows).
  - Early stopping at **1,533 trees**, holdout AUC 0.99975.

**Threshold selection method:** macro-F0.5 optimization over a 0.05–0.95 grid on the 20,000-S1 fold-0 development subset, with the one-target → one-S1 rule applied, gave **p ≥ 0.69**. After thresholding, each target goes only to the S1 that gives it the highest probability, applied globally over the whole test set. An S1 with no pair above the threshold gets an empty list.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.92813** on the 20,000-S1 fold-0 development subset (v1: 0.92694 on the same S1). Pair precision is 0.9878 (v1 0.9847) and pair recall 0.8512 (v1 0.8545). 7.75% of dev S1 are predicted empty. The v1 submission scored 0.927 on the public leaderboard, in line with its dev score. We make no leaderboard claim for v2.

  | Threshold | 0.40 | 0.50 | 0.60 | 0.63 | 0.69 | 0.75 | 0.80 | 0.85 |
  |---|---|---|---|---|---|---|---|---|
  | v1 macro F0.5 | 0.92176 | 0.92469 | 0.92637 | 0.92694 | — | 0.92640 | 0.92543 | 0.92342 |
  | v2 macro F0.5 | 0.92317 | 0.92624 | 0.92750 | 0.92794 | **0.92813** | 0.92779 | 0.92672 | 0.92512 |

- **Paired comparison (same 20k dev S1):** mean per-S1 F0.5 difference of +0.00119, 95% bootstrap CI [+0.00037, +0.00197]. 416 S1 improve, 450 get worse and 19,134 are unchanged. More S1 get worse than better, yet the mean still rises, because the per-S1 changes are not the same size. The gain is small and statistically supported, but it is not large.
- **Where errors come from:** the top-50 lists contain 89.0% of all true pairs, and the matcher recovers 85.1%. The remaining gap to the top-50 oracle (0.953) is mostly missed pairs, while false merges stay rare (precision 0.988).
- **Common false positives (wrong merges):** we did not do a manual audit of individual errors in the time available. By design, the riskiest pairs are near-identical names whose address evidence is missing or ambiguous (chain outlets, common names). The house/unit/postcode conflict features and the one-target → one-S1 rule target exactly this case.
- **Common false negatives (missed matches):** quantitatively, most missed pairs never reach the top 50 (about 11% of true pairs). The expected causes are name changes that share no token or key with any retrieval channel (DBA/trade names, heavy abbreviation, transliterations beyond the skeleton key), made worse when the address is missing.
- **Test-set output (v2):**
  - @@MATCHED@@ predicted pairs: @@S2@@ in S2 and @@S3@@ in S3.
  - @@S1_WITH@@ S1 with at least one match, and @@S1_EMPTY@@ S1 predicted empty.
  - The official validator reported @@VALIDATOR@@.

---

## 6. Conclusion
A precision-first cascade of gradient-boosted trees reaches macro F0.5 ≈ 0.928 on held-out data. v2 comes from a controlled, evidence-based change: more Stage-2 training data, with a gain confirmed by a paired bootstrap on identical dev S1. The analysis also shows the limits. The top-50 oracle is 0.953, and even all retrieved candidates allow only about 0.98, so large further gains need better Stage-1 recall at a fixed candidate count (a larger cap and the full ranker) rather than a bigger candidate set or threshold tuning.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains:
- `src/ber/`, with these submodules:
  - `io` and `normalize`: raw and normalized caches.
  - `tokens` and `retrieval`: per-country inverted indexes, channels, the union and the Indic key.
  - `cv`: stratified S1 folds.
  - `ranker`: Stage-1 features and training utilities.
  - `matcher`: Stage-2 features and the decision rule.
  - `predict`: end-to-end inference.
  - `rescore`: v2 re-scoring of the v1 Stage-1 top-50 with a new Stage-2 model.
- `experiments/`:
  - `e02`–`e04`: retrieval, Stage 1, Stage 2 (`e04 --save-rows`).
  - `e05`: Stage-2 variants.
  - `e06`: paired bootstrap.
- `tests/`, `README.md` (sections 9–10 give the exact commands) and `requirements.txt`.

**How the v2 files were produced:**
```
cd code/business_entity_resolution/src
python ../experiments/e04_matcher.py --train 100000 --valid 20000 --out-name v2_t100000_v20000 --save-rows
python -m ber.rescore --ref-run full --run-name v2 --stage2-dir ../../../work/experiments/matcher/v2_t100000_v20000
python -m ber.rescore --ref-run full --run-name v2 --stage2-dir ../../../work/experiments/matcher/v2_t100000_v20000 --finalize --out ../../../output_v2 --validate
```
`ber.rescore` recomputes retrieval, the cap, the Stage-1 features and the 62 Stage-2 features. It takes the Stage-1 top-50 (pairs, raw scores, ranks) from the v1 run instead of re-running the Stage-1 LightGBM. With the v1 Stage-2 model it reproduces v1's probabilities exactly (maximum difference 0). A single pass of `python -m ber.predict --split test --run-name v2_direct --stage2-dir <v2 model dir> --out ../../../output_v2 --validate` computes the same result from scratch.

**Output validation:** the official `validate_submission.py --check-ids` reported @@VALIDATOR@@. Independent checks: every test S1 appears exactly once in each file, every target ID is a valid S2-/S3- test ID, there are no duplicate pairs, no target is matched to more than one S1, and every predicted pair is in `candidate_pairs.tsv`.

### B. Additional Results
| Experiment | Candidates | Model | Threshold | Dev macro F0.5 | Precision | Recall | Decision |
|---|---|---|---|---|---|---|---|
| v1 | G+Indic, cap 500, top 50 | Stage 2 on 60k S1, 1,336 trees | 0.63 | 0.92694 | 0.9847 | 0.8545 | baseline |
| Threshold sweep (v1) | same | same | 0.40–0.85 | ≤ 0.92694 | 0.972–0.993 | 0.866–0.832 | keep 0.63 |
| Top-K 100 / 200 (Stage-1 holdout) | 2× / 4× candidates | — | — | oracle 0.962 / 0.964 (vs 0.955) | — | top-K recall 0.903 / 0.907 (vs 0.892) | rejected (small gain, much larger candidate set) |
| Cap 300 (Stage-1 holdout) | top 50 | — | — | oracle 0.951 | — | 0.882 | rejected |
| Cap 2000, 1,662 trees (Stage-1 holdout) | top 50 | — | — | oracle 0.965 | — | 0.910 | promising; too slow for tonight |
| **v2** | same as v1 | **Stage 2 on 100k S1, 1,533 trees** | **0.69** | **0.92813** | 0.9878 | 0.8512 | **selected** (+0.00119, CI [+0.00037, +0.00197]) |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
