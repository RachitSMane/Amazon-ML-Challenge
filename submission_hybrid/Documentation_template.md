# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CloundForge  
**Team Members:** Rachit Mane, Prithvi.W, Pranav .V.Mennon, Akash.R  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A three-stage pipeline: IDF-weighted sparse retrieval over normalized names, addresses and two exact name keys (including a phonetic skeleton key for Indic scripts) generates candidates. A LightGBM ranker re-ranks them, and a second LightGBM classifier with 62 string, number and ranking features decides the matches at a precision-oriented threshold. A global rule assigns each Source-2/3 record to at most one Source-1 record. **This second submission** comes from a validation study that identified Stage-1's candidate cap as the main bottleneck. It uses **v3**:
- the Stage-1 cap is raised from 500 to 1000;
- 200 candidates per S1 are passed to Stage 2;
- Stage 2 is retrained on that distribution.

On the same 20,000 held-out development S1, v3 scores **macro F0.5 0.93766**, against 0.92694 for our first submission (v1). A full v3 test pass needs about 10 hours, so v3 scored the test S1 it could before the deadline (@@V3_S1@@ S1, @@V3_PCT@@%). Every other S1 keeps its validated v1 prediction.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Precision-heavy, singleton-aware metric.** F0.5 is macro-averaged per S1, and an S1 with no true match scores 1.0 only if nothing is predicted. Wrong merges are therefore more costly than missed links, and predicting "empty" has to be a real option for every record.
- **Name noise:** legal-form variants (Pvt Ltd / Private Limited, Inc / Corporation, SARL / SAS), `&` vs and/et, punctuation and dotted initialisms, French elisions, accents, word order, typos, and Indic-script names with transliteration and zero-width-joiner differences.
- **Address noise:** `<NULL>` placeholders appear as individual comma-separated components, abbreviations vary, house numbers are zero-padded differently across sources, and PIN/ZIP codes or whole components are missing.
- **Open country set:** the test set adds France, which is absent from training. Every component is country-agnostic, and country only partitions the target index.
- **Scale:** 1.73 M test S1 against 9.97 M S2+S3 targets. This requires vectorized sparse retrieval with a bounded candidate set per S1.

### 2.2 Solution Strategy
**Approach Type:** Blocking (sparse IDF retrieval) + learned ranker + learned classifier + global assignment constraint  
**Core Innovation:**
- A union of capped IDF retrieval channels plus an Indic phonetic-skeleton key.
- A two-stage LightGBM ranker and matcher, where the Stage-1 scores and ranks become Stage-2 features.
- An evidence-driven choice of the Stage-1 cap and top-K.
- Exact, resumable, label-free block-wise inference.

All model, cap, top-K and threshold selection uses deterministic S1-level cross-validation. The folds are stratified by country × number of true matches, and the fixed 20,000-S1 fold-0 development subset is used only for evaluation and threshold choice. The leaderboard was not used to select anything.

### 2.3 How v3 was chosen (validation evidence)
| Held-out data | v1 | v2 | **v3** |
|---|---|---|---|
| Stage 1 | 400 trees, cap 500, top 50 | same | 400 trees, **cap 1000, top 200** |
| Stage 2 | 60k S1, 1,336 trees | 100k S1, 1,533 trees | 60k S1, 523 trees |
| Top-K pair recall (dev) / oracle F0.5 | 0.890 / 0.9534 | same | **0.920 / 0.9685** |
| Threshold (dev optimum, 0.01 grid) | 0.63 | 0.69 | **0.66** |
| Precision / recall (dev) | 0.9847 / 0.8545 | 0.9878 / 0.8512 | 0.9837 / **0.8746** |
| **Macro F0.5 (dev)** | 0.92694 | 0.92813 | **0.93766** |
| Empty S1 (dev) | 7.60% | 7.75% | 6.92% |

- **Threshold:** v1's threshold was already dev-optimal, and its curve is flat.
- **v2:** retraining Stage 2 on more data helped only slightly (+0.00119, paired-bootstrap 95% CI [+0.00037, +0.00197]).
- **Stage-1 sweep:** on 2,000 holdout S1, we compared 400, 800, 1,200 and 1,662 trees × caps of 300, 500, 750 and 1000 × top-K of 20–200. Retrieval returns 95% of true pairs, but the union-rank cap of 500 kept only 91%.
  - Raising the cap to 1000 lifted top-50 recall from 0.892 to 0.900.
  - Using all 1,662 trees added only +0.3 points, at 3× the cost.
  - Top-200 at cap 1000 reached 0.919 pair recall and 0.9691 oracle F0.5.
- **v3 result:** retraining Stage 2 on the cap-1000 / top-200 candidates turned that recall into +1.07 points of macro F0.5 over v1. It is better at every threshold from 0.40 to 0.85, and its worst point there (0.931) is above v1's best (0.927).

---

## 3. Candidate Generation (Blocking)
Normalization (`ber.normalize`) is deterministic: NFKC, casefold, invisible-character removal, and placeholder removal at the component level. Legal forms are stripped using explicit per-country lists. Addresses get number canonicalization and abbreviation expansion. Targets are indexed per country as inverted CSR matrices, and each channel scores a query as the IDF sum of the tokens it shares with a target. Tokens above a channel's document-frequency cap are suppressed for that channel.

- **Blocking keys used ("G+Indic" union):**
  - C1: name tokens, DF cap 10,000.
  - C2: address tokens, DF cap 1,000.
  - K: exact strong-name key (legal forms stripped, tokens sorted).
  - Indic key: exact sorted phonetic-skeleton key of the stripped name, matched against Indic or mixed-script targets.

  Channel scores are merged per (S1, target), and candidates are ordered by union IDF score.
- **Candidate pairs generated:**
  - Retrieval returns about 4,000 targets per S1.
  - Stage 1 re-ranks the best **1000** by union rank (v3) or 500 (v1).
  - It keeps the top **200** (v3) or 50 (v1) per S1.
  - `candidate_pairs.tsv` holds, for each S1, exactly the list scored by the Stage-2 model that decided that S1: @@CAND_PAIRS@@ pairs for 1,732,544 test S1.
- **How you ensured true matches were not lost:** every channel, union, cap and top-K was measured on held-out S1 (pair recall, S1 recall, oracle F0.5):
  - G union: 0.931 pair recall.
  - With the Indic key: 0.953.
  - v3 top-200: 0.920 on the 20k dev S1.

---

## 4. Matching Model

**Features used (62; unchanged schema, same order in training and inference):**
- Name features: per-channel IDF score, rank, query/target token-coverage fractions and weighted Jaccard (name, strong key, Indic key); RapidFuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the normalized and legal-form-stripped name; equality of legal form; equality of the first stripped token; token counts; the query's maximum token IDF; key document frequencies.
- Address features: per-channel IDF score, rank, coverage fractions and weighted Jaccard; RapidFuzz ratio, token-set and partial ratio; token-set similarity over numbers only; agreement and conflict flags for house number, unit number and postcode; missing-address flags.
- Other: union score, rank and relative score; number of channels; log candidate count; target source; query and target scripts; and the Stage-1 raw score, rank, gap to the best, best score, gap between the 1st and 2nd scores, and number of near-top candidates.
- **Importance (gain, v3):** Stage-1 score 0.66, Stage-1 rank 0.11, address token-set similarity 0.08, address number token-set 0.04, name partial ratio 0.02.

**Model type:** LightGBM gradient-boosted trees only (MIT license; no neural or pretrained model).
- **Stage 1:** `model_GI_unweighted.txt`, trained on 100k S1 from folds 1–4, using its first 400 trees.
- **Stage 2 (v3):** binary LightGBM (learning rate 0.05, 63 leaves, min 200 rows per leaf, feature fraction 0.9, bagging 0.8, L2 1.0, deterministic).
  - Trained on every top-200 candidate of 60,000 fold-1–4 S1, excluding the Stage-1 training S1: 10.3 M rows, 1.7% positive.
  - A 10% S1-hash early-stopping holdout.
  - 523 trees, holdout AUC 0.99987.

**Threshold selection method:** macro-F0.5 maximization over a 0.05–0.95 grid (step 0.01) on the 20,000 dev S1, with the one-target → one-S1 rule applied. v3 uses **0.66** and v1 uses 0.63.

**Hybrid decision rule (this submission):**
- Each test block of 500 S1 uses v3's checkpoint if v3 scored it, and otherwise v1's.
- A pair is kept if its probability is at or above the threshold of the model that scored it.
- The one-target → one-S1 rule is then applied **globally over the whole test set**: each target goes to its highest-probability S1.
- With no v3 blocks, the assembler reproduces the v1 submission files byte-for-byte.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.93766** for v3 on the 20,000 fold-0 dev S1 (precision 0.9837, recall 0.8746, 6.92% of S1 predicted empty), vs 0.92694 for v1 on the same S1. The v1 submission scored 0.927 on the public leaderboard, in line with its dev score. The hybrid as a whole was not evaluated on dev, because it mixes two models on test only, and we make no leaderboard claim for it.

  | Threshold | 0.40 | 0.50 | 0.60 | 0.63 | 0.66 | 0.70 | 0.80 | 0.85 |
  |---|---|---|---|---|---|---|---|---|
  | v1 macro F0.5 | 0.92176 | 0.92469 | 0.92637 | 0.92694 | 0.92675 | 0.92663 | 0.92543 | 0.92342 |
  | v3 macro F0.5 | 0.93121 | 0.93468 | 0.93692 | 0.93738 | **0.93766** | 0.93718 | 0.93559 | 0.93289 |

- **Where errors come from:** v3's top-200 lists contain 92.0% of all true pairs (v1: 89.0%), and its matcher recovers 87.5% (v1: 85.5%). The remaining gap to the top-200 oracle (0.968) is mostly missed pairs, while precision stays at about 0.984.
- **Common false positives (wrong merges):** we did not do a manual audit of individual errors in the time available. By design, the riskiest pairs are near-identical names whose address evidence is missing or ambiguous (chain outlets, common names). The house/unit/postcode conflict features and the one-target → one-S1 rule target this case.
- **Common false negatives (missed matches):** quantitatively, about 8% of true pairs are still outside v3's top 200. The expected causes are name changes that share no token or key with any retrieval channel, especially when the address is missing.
- **Test-set output (hybrid):**
  - @@V3_S1@@ S1 (@@V3_PCT@@%) decided by v3 and @@V1_S1@@ by v1.
  - @@MATCHED@@ predicted pairs: @@S2@@ in S2 and @@S3@@ in S3.
  - @@REMOVED@@ above-threshold pairs removed by the one-target → one-S1 rule.
  - @@S1_EMPTY@@ S1 predicted empty.
  - The official validator reported @@VALIDATOR@@.

---

## 6. Conclusion
Measuring where the score is lost showed that the Stage-1 cap, not the classifier or the threshold, was the main bottleneck. Doubling the cap, passing 200 candidates to Stage 2 and retraining Stage 2 on that distribution raised held-out macro F0.5 from 0.927 to 0.938 with the same features and models. The remaining limits are test-time cost (the Stage-1 ranker on 1000 candidates dominates) and the roughly 8% of true pairs that retrieval and ranking still miss. With more time, the next steps are the full v3 pass and cheaper Stage-1 scoring.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains:
- `src/ber/`, with these submodules:
  - `io` and `normalize`: raw and normalized caches.
  - `tokens` and `retrieval`: indexes, channels, the union and the Indic key.
  - `cv`: stratified folds.
  - `ranker`: Stage-1 features.
  - `matcher`: Stage-2 features and the decision rule.
  - `predict`: end-to-end inference.
  - `score_blocks`: block worker with `--stage2-dir`, `--run-dir` and `--until`.
  - `assemble_hybrid`: this submission's assembly.
  - `rescore` and `assemble_partial`: helpers.
- `experiments/`:
  - `e02`–`e04`: retrieval, Stage 1, Stage 2 (`e04` takes `--cap`, `--top-k` and `--out-dir`).
  - `e03_ranker_fast_eval`: the Stage-1 sweep (`--k-values`).
  - `e05`: Stage-2 variants.
  - `e06`: paired bootstrap.
- `tests/`, `README.md` (sections 9–11 give the exact commands) and `requirements.txt`.

**How the hybrid files were produced:**
```
cd code/business_entity_resolution/src
python ../experiments/e04_matcher.py --train 60000 --valid 20000 --cap 1000 --top-k 200 --out-name t60000_v20000 --out-dir experiments/stage1_400_cap1000_top200/t60000_v20000
python -m ber.score_blocks --ref-run full --run-name v3 --run-dir experiments/stage1_400_cap1000_top200/test_run --stage2-dir ../../../work/experiments/stage1_400_cap1000_top200/t60000_v20000 --start 519 --stop 2139 --until 22:55
python -m ber.assemble_hybrid --primary experiments/stage1_400_cap1000_top200/test_run --primary-stage2 ../../../work/experiments/stage1_400_cap1000_top200/t60000_v20000 --fallback inference/test/full --fallback-stage2 ../../../work/experiments/matcher/full_t60000_v20000 --out ../../../work/experiments/stage1_400_cap1000_top200/output_hybrid --validate
```
The v1 part comes from `python -m ber.predict --split test --run-name full --out ../../../output --validate`.

**Output validation:** the official `validate_submission.py --check-ids` reported @@VALIDATOR@@. Independent checks:
- Every test S1 appears exactly once in each file.
- Every target ID is a valid S2-/S3- test ID.
- There are no duplicate pairs.
- No target is matched to more than one S1.
- Every predicted pair is in `candidate_pairs.tsv`.

### B. Additional Results
| Experiment | Trees | Cap | Top-K | Pair recall | Oracle F0.5 | Dev macro F0.5 | Decision |
|---|---|---|---|---|---|---|---|
| v1 | 400 | 500 | 50 | 0.892 (holdout) / 0.890 (dev) | 0.955 / 0.953 | 0.92694 | first submission |
| v2 (Stage 2 on 100k) | 400 | 500 | 50 | same | same | 0.92813 | superseded |
| Stage-1 trees | 1662 | 500 | 50 | 0.896 | 0.958 | — | poor value (3× cost) |
| Stage-1 cap | 400 | 750 | 50 | 0.897 | 0.958 | — | — |
| Stage-1 cap | 400 | 1000 | 50 | 0.900 | 0.959 | — | — |
| Stage-1 top-K | 400 | 1000 | 100 | 0.911 | 0.965 | — | — |
| **v3** | 400 | 1000 | 200 | 0.919 (holdout) / 0.920 (dev) | 0.969 / 0.968 | **0.93766** | **used for this submission** |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
