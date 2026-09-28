"""Union of channel outputs into one deduplicated candidate list per query.

A ``Strategy`` names a set of channels and an optional union-level ``top_k``. For each
(query, target) pair found by any channel, the union keeps:

* ``score``: the sum of the channel scores (each channel contributes its IDF overlap);
* ``mask``: a bit per channel (bit i = found by the i-th channel of the strategy);
* ``rank``: the position of the pair within its query by combined score (0 = best, ties
  broken by target index), so a union-level top-K is ``rank < k``.

Duplicates across channels are merged, never counted twice.
"""

from dataclasses import dataclass

import numpy as np

from .sparse_topk import rank_rows


@dataclass(frozen=True)
class Strategy:
    name: str
    channels: tuple
    top_k: int = None


def union_candidates(parts, n_queries, top_k=None):
    """Merge ``parts`` = [(q, t, score), ...] (one per channel) into unique ranked pairs.

    ``q`` are query rows in ``[0, n_queries)`` and ``t`` target indices (both int32).
    Returns a dict of arrays ``q, t, score, mask, rank`` sorted by query, then rank.
    """
    if len(parts) > 8:
        raise ValueError("at most 8 channels per strategy (uint8 mask)")
    q_all = np.concatenate([p[0] for p in parts]).astype(np.int64)
    t_all = np.concatenate([p[1] for p in parts]).astype(np.int64)
    s_all = np.concatenate([p[2] for p in parts]).astype(np.float64)
    keys, inverse = np.unique((q_all << 32) | t_all, return_inverse=True)
    score = np.bincount(inverse, weights=s_all, minlength=len(keys)).astype(np.float32)
    mask = np.zeros(len(keys), dtype=np.uint8)
    offset = 0
    for bit, part in enumerate(parts):
        n = len(part[0])
        mask[inverse[offset:offset + n]] |= np.uint8(1 << bit)
        offset += n
    q = (keys >> 32).astype(np.int32)
    t = (keys & 0xFFFFFFFF).astype(np.int32)
    order, rank = rank_rows(q, t, score, n_queries)
    out = {"q": q[order], "t": t[order], "score": score[order], "mask": mask[order], "rank": rank.astype(np.int32)}
    if top_k is not None:
        keep = out["rank"] < top_k
        out = {k: v[keep] for k, v in out.items()}
    return out
