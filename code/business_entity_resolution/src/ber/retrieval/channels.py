"""Retrieval channels: which field, which DF cap, which per-query top-K.

A channel scores every target of the query's country that shares at least one
non-suppressed token with the query, using the sum of the shared tokens' IDF. Tokens whose
document frequency (in that country's target pool) exceeds ``df_cap`` are suppressed for
this channel only; with ``top_k`` set, only the ``top_k`` best-scoring targets are kept.

Initial channels (Phase 2):

* ``C1``: tokens of ``name_fold`` (normalized, accent-folded name);
* ``C2``: tokens of ``address_norm``;
* ``C3``: tokens of ``name_stripped`` (legal forms removed);
* ``K``:  exact ``name_key`` (legal-stripped, sorted tokens; "strong name"), no cap by default.
"""

from dataclasses import dataclass

import numpy as np

from ..tokens import cap_query_matrix
from .sparse_topk import score_block, topk_per_row


@dataclass(frozen=True)
class Channel:
    """One retrieval channel configuration."""

    name: str
    field: str
    df_cap: int = None
    top_k: int = None


def C1(df_cap=1000, top_k=None):
    return Channel(f"name_fold<={df_cap}" + (f"@{top_k}" if top_k else ""), "name_fold", df_cap, top_k)


def C2(df_cap=1000, top_k=None):
    return Channel(f"address<={df_cap}" + (f"@{top_k}" if top_k else ""), "address_norm", df_cap, top_k)


def C3(df_cap=1000, top_k=None):
    return Channel(f"name_stripped<={df_cap}" + (f"@{top_k}" if top_k else ""), "name_stripped", df_cap, top_k)


def K(df_cap=None, top_k=None):
    return Channel("strong_name" + (f"<={df_cap}" if df_cap else ""), "name_key", df_cap, top_k)


class ChannelRunner:
    """Runs channels for one country's query block, caching DF-capped query matrices."""

    def __init__(self, country_pool, query_matrices):
        self.pool = country_pool
        self.queries = query_matrices  # field -> uncapped CSR (queries × vocab)
        self._capped = {}

    def capped(self, field, cap):
        key = (field, cap)
        if key not in self._capped:
            self._capped[key] = cap_query_matrix(self.queries[field], self.pool.fields[field].df, cap)
        return self._capped[key]

    def run(self, channel, start, stop):
        """Candidates of query rows ``[start, stop)``.

        Returns ``(q, t, score, rank)``: query rows relative to ``start`` (int32), global
        pool target indices (int32), IDF-sum scores (float32) and in-query ranks (int32).
        """
        block = self.capped(channel.field, channel.df_cap)[start:stop]
        m = score_block(block, self.pool.fields[channel.field].inverted)
        rows, cols, scores, rank = topk_per_row(m, channel.top_k)
        return rows, self.pool.target_idx[cols].astype(np.int32), scores, rank
