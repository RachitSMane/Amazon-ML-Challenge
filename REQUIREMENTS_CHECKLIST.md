# Verified Requirements Checklist: Amazon ML Challenge 2026

Sources, re-read on 2026-09-25:

- **[R]** `student_resource/README.md`: official problem statement
- **[C]** `6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf`: the problem statement plus an **update banner**
- **[G]** `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`: organiser guidelines
- **[V]** `student_resource/utils/validate_submission.py`: the official validator

Where `PLAN.md` conflicts with these, **these win**.

## A. Official requirements

### Task and data
- [ ] For **every** test S1 entity, output all matching S2/S3 entity IDs. Zero, one or many matches are possible. [R]
- [ ] Source 1 is the deduplicated reference source. [R]
- [ ] Every file is **TSV**, read with `sep="\t"`. Addresses and ID lists contain commas. [R]
- [ ] Source columns: `entity_id` (prefix `S1-`, `S2-` or `S3-`), `business_name`, `business_address`, `country`. There is no source column; the prefix gives the source. [R]
- [ ] **Country is an open set.** Train has US and India; **test adds France**. Don't hard-code, filter or one-hot `{US, India}`. Every France entity must appear in the output. [R]

### Output files
- [ ] `matching_results.tsv`, header exactly `source1_entity_id<TAB>matched_entity_ids`. **This is the only file scored.** [R]
- [ ] `candidate_pairs.tsv`, header exactly `source1_entity_id<TAB>candidate_entity_ids`. It must be the **final** candidate set the model runs inference on, not an earlier blocking pass. [R]
- [ ] Both files: one row per test S1 entity; an empty list for no match; only S2-/S3- IDs; no duplicates in a list; no duplicate S1 rows; ID lists comma-separated with **no quoting**. [R][V]
- [ ] The matches should be a subset of the candidates. The validator warns if they aren't. [R][V]

### Candidate set size (update banner) [C]
- [ ] Blocking must scale. All-pairs comparison "is not an option".
- [ ] `candidate_pairs.tsv` and the code that produces it are **reviewed for the final ranking**, alongside the score.
- [ ] **A smaller candidate set per S1 entity is ranked higher**, beyond the public/private leaderboard.

### Metric [R]
- [ ] **Macro F0.5** per S1 entity: F0.5 = 1.25·P·R / (0.25·P + R), averaged over **all** S1 entities.
- [ ] Singletons count: predicting empty scores 1.0, and predicting anything scores 0.0.
- [ ] The public leaderboard uses a subset of the test set. **The final ranking uses the private leaderboard.** Submit predictions for the full test set.

### Model and data rules [R]
- [ ] The final model must be **MIT or Apache-2.0 licensed** and have **≤ 8B parameters**.
- [ ] **No external lookups:** no ER APIs, government registries, geocoding APIs or internet data augmentation. Breaking this means **disqualification**. Use only the provided training data.

### Submission process [R][G]
- [ ] **At most 5 leaderboard uploads per day** during the window. [G]
- [ ] Window: **25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST**. [G]
- [ ] Keep a **version history of all submissions**. Shortlisting is based on the submitted solutions. [G]
- [ ] A valid upload shows `SCORED` with an F0.5; files that fail validation are not evaluated. [R]
- [ ] Final zip `<team_name>_submission.zip`: [R]
  - [ ] `output/matching_results.tsv` and `output/candidate_pairs.tsv`
  - [ ] `code/business_entity_resolution/` containing `src/`, a `README.md` (end-to-end reproduction) and a pinned `requirements.txt`. **It must regenerate both outputs from the data using only this folder.**
  - [ ] the filled-in `Documentation_template.md` (`.md` or `.pdf`)
- [ ] The methodology document covers: methodology, candidate generation/blocking, model architecture and features, and anything else relevant. [R][G]
- [ ] The source code has "proper comments describing the functions". [G]
- [ ] One login per participant, from a desktop or laptop only. [G]

### Validator behaviour [V]
- [ ] Run it from `student_resource/`. `PASS` exits 0; `FAIL` exits 1 with a numbered list.
- [ ] It checks the header (case-insensitive), the tab/CSV detection, duplicate rows, duplicate IDs within a list, S1 self-matches, bad prefixes, and missing or extra S1 rows.
- [ ] The ID-existence check is only done with `--check-ids`, which uses a lot of memory.
- [ ] It does **not** check your score, the candidate set size, or an S2/S3 ID appearing in several rows.

## B. Conflicts and ambiguities (ask via the query form)

| # | Conflict | Working rule until clarified |
|---|---|---|
| 1 | Nonexistent IDs "will be rejected" [R] vs "only lowers your score, never rejects" [V] | Never output an ID that isn't in the test files; run `--check-ids` before the final upload |
| 2 | Document "no page limit" [R] vs "1–2 page document" [G] | Write a thorough template and keep a 1–2 page summary at the top |
| 3 | How candidate set size is weighed against F0.5 [C] | Minimise candidates per S1 while keeping the recall ceiling high; report both |
| 4 | Whether pretrained open-source models (e.g. embeddings) count as "external data" | Allowed only if Apache/MIT, ≤ 8B, run locally and not used to look up entities; confirm |

## C. Team assumptions (not official, must be verified)

| Assumption | Status |
|---|---|
| Each S2/S3 record matches at most one S1 record | **Unverified.** Check `ground_truth.ONE_MATCH_ASSUMPTION_…` from `analyze_dataset.py` |
| Country labels are reliable enough to block on | Unverified: check the country distributions and `true_pairs.same_country` |
| Postcodes are present often enough to block on | Unverified: check `patterns_share.address_has_postcode_like` and `postcode_equal_given_both_present` |
| A GBDT over string-similarity features is enough | Design choice, to be decided after the EDA |
| Candidate budget K≈10–30 (old PLAN.md) | **Superseded** by [C]; set K from measured recall and set-size curves |
