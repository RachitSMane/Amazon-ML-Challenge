"""Candidate retrieval (blocking): sparse token channels, unions and streaming evaluation.

* ``sparse_topk``: chunked ``queries @ inverted`` scoring and deterministic per-row top-K.
* ``channels``: channel configurations (field, DF cap, top-K) and the per-country runner.
* ``union``: deduplicated union of channels with combined score, channel mask and rank.
* ``evaluate``: pair recall, S1 recall and oracle F0.5 without storing candidates.

This is the internal retrieval pool. The final candidate set (``candidate_pairs.tsv``) is
produced later, after cheap pruning; nothing here writes it.
"""
