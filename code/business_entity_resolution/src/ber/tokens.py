"""Deterministic tokenization, per-pool/per-country vocabularies, DF/IDF and inverted indices.

Inputs are the normalized columns of ``work/cache/norm_v1`` (never modified here):

* ``tokens`` fields (``name_fold``, ``name_stripped``, ``address_norm``): the column is split on
  single spaces (Phase 1 already collapsed whitespace); duplicates within one record count once.
* ``key`` fields (``name_key``): the whole value is one token (exact-key blocking).
  Empty values produce no token.

A **target pool** is all S2+S3 records of one split (train or test), in the order of
``io.load_targets`` (S2 rows, then S3). It is partitioned by country, and every
(pool, country, field) gets its own ``FieldIndex``:

* ``vocab``: the distinct tokens, sorted (so ids do not depend on row order);
* ``df``: number of targets of that country containing the token (int32);
* ``idf``: ``log(n_targets / df)`` (float32);
* ``inverted``: CSR matrix ``vocab × targets`` with 1.0 entries (int32 indices), i.e. an
  inverted index, so ``queries @ inverted`` scores every target sharing a token.

Country partitions keep each matrix small and make country a hard blocking key (no true
pair crosses countries). S1 records are **queries**: ``query_matrix`` maps their tokens into
a country's vocabulary (out-of-vocabulary tokens cannot retrieve anything and are dropped)
with the token IDF as value. DF caps are applied to queries only (``cap_query_matrix``), so
no token is ever removed from the stored data.
"""

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from . import io
from .normalize import cache as norm_cache

FIELD_MODES = {"name_fold": "tokens", "name_stripped": "tokens", "address_norm": "tokens", "name_key": "key"}


def tokenize(values, mode):
    """Split an Arrow string array into ``(doc, token)`` pairs; empty tokens are dropped.

    Returns ``doc`` (int64 row positions) and ``tokens`` (Arrow string array), in row order.
    """
    values = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if pa.types.is_null(values.type):  # e.g. pa.array([]) for an empty partition
        values = values.cast(pa.string())
    if mode == "key":
        doc = np.arange(len(values), dtype=np.int64)
        tokens = values
    elif mode == "tokens":
        lists = pc.split_pattern(values, " ")
        doc = pc.list_parent_indices(lists).to_numpy().astype(np.int64)
        tokens = pc.list_flatten(lists)
    else:
        raise ValueError(f"unknown tokenization mode {mode!r}")
    keep = pc.not_equal(tokens, "")
    return doc[keep.to_numpy(zero_copy_only=False)], tokens.filter(keep)


def _unique_pairs(rows, cols, n_cols):
    """Sorted unique (row, col) pairs of two int arrays (by row, then col)."""
    n_cols = max(int(n_cols), 1)
    keys = np.unique(rows.astype(np.int64) * n_cols + cols)
    return (keys // n_cols).astype(np.int64), (keys % n_cols).astype(np.int32)


@dataclass
class FieldIndex:
    """Vocabulary, document frequencies and inverted index of one field in one country."""

    vocab: pa.Array
    df: np.ndarray
    idf: np.ndarray
    inverted: sp.csr_matrix
    n_docs: int

    @property
    def n_tokens(self):
        return len(self.vocab)

    def nbytes(self):
        inv = self.inverted
        return int(inv.data.nbytes + inv.indices.nbytes + inv.indptr.nbytes + self.df.nbytes
                   + self.idf.nbytes + self.vocab.nbytes)


def build_field_index(values, mode):
    """Build a ``FieldIndex`` from the values of one field for one country's targets."""
    n_docs = len(values)
    doc, tokens = tokenize(values, mode)
    vocab = pc.unique(tokens)
    vocab = vocab.take(pc.sort_indices(vocab))
    token_id = pc.index_in(tokens, value_set=vocab).to_numpy(zero_copy_only=False).astype(np.int64)
    doc, token_id = _unique_pairs(doc, token_id, len(vocab))
    df = np.bincount(token_id, minlength=len(vocab)).astype(np.int32)
    idf = np.log(n_docs / np.maximum(df, 1)).astype(np.float32)
    # Inverted index: rows = tokens, columns = documents (sort by token, then doc).
    order = np.lexsort((doc, token_id))
    indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    inverted = sp.csr_matrix(
        (np.ones(len(order), dtype=np.float32), doc[order].astype(np.int32), indptr),
        shape=(len(vocab), n_docs),
    )
    return FieldIndex(vocab=vocab, df=df, idf=idf, inverted=inverted, n_docs=n_docs)


def query_matrix(values, mode, index):
    """CSR ``queries × vocab`` with the IDF of each in-vocabulary query token (float32)."""
    n_q = len(values)
    doc, tokens = tokenize(values, mode)
    token_id = pc.index_in(tokens, value_set=index.vocab)
    known = pc.is_valid(token_id).to_numpy(zero_copy_only=False)
    doc = doc[known]
    token_id = token_id.filter(pc.is_valid(token_id)).to_numpy().astype(np.int64)
    doc, token_id = _unique_pairs(doc, token_id, index.n_tokens)
    indptr = np.zeros(n_q + 1, dtype=np.int64)
    np.cumsum(np.bincount(doc, minlength=n_q), out=indptr[1:])
    return sp.csr_matrix((index.idf[token_id], token_id.astype(np.int32), indptr),
                         shape=(n_q, index.n_tokens))


def cap_query_matrix(q, df, cap):
    """Copy of ``q`` without the entries whose token DF exceeds ``cap`` (None = no cap)."""
    if cap is None:
        return q
    keep = df[q.indices] <= cap
    rows = np.repeat(np.arange(q.shape[0]), np.diff(q.indptr))
    counts = np.bincount(rows[keep], minlength=q.shape[0])
    indptr = np.zeros(q.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return sp.csr_matrix((q.data[keep], q.indices[keep], indptr), shape=q.shape)


# --------------------------------------------------------------------------- pools

@dataclass
class CountryPool:
    """The targets of one country in a pool: their global pool indices and field indices."""

    country: str
    target_idx: np.ndarray  # int32, ascending global positions in the pool
    fields: dict = field(default_factory=dict)


@dataclass
class TargetPool:
    """All S2+S3 targets of one split, partitioned by country."""

    split: str
    entity_id: pa.Array
    source: np.ndarray       # int8 (2 or 3), per global target
    name_script: pa.Array
    address_missing: np.ndarray
    countries: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.entity_id)

    def nbytes(self):
        return sum(f.nbytes() + cp.target_idx.nbytes for cp in self.countries.values() for f in cp.fields.values())


def read_normalized(cfg, split, sources, columns):
    """Concatenate the normalized tables of ``sources`` (in that order) for ``split``."""
    for source in sources:
        if not norm_cache.is_norm_fresh(cfg, io.source_key(split, source)):
            raise RuntimeError(f"normalized cache stale for {split}_source{source}")
    tables = [pq.read_table(norm_cache.norm_file(cfg, io.source_key(split, s)), columns=columns)
              for s in sources]
    return pa.concat_tables(tables).combine_chunks()


def build_target_pool(cfg, split, fields=tuple(FIELD_MODES), log=None):
    """Load the S2+S3 targets of ``split`` and index ``fields`` per country."""
    columns = ["entity_id", "country", "source", "name_script", "address_missing"] + list(fields)
    table = read_normalized(cfg, split, (2, 3), columns)
    pool = TargetPool(
        split=split,
        entity_id=table["entity_id"].combine_chunks(),
        source=table["source"].to_numpy(),
        name_script=table["name_script"].combine_chunks(),
        address_missing=table["address_missing"].to_numpy(zero_copy_only=False),
    )
    country = table["country"].combine_chunks()
    for c in sorted(pc.unique(country).to_pylist()):
        idx = np.flatnonzero(pc.equal(country, c).to_numpy(zero_copy_only=False)).astype(np.int32)
        cpool = CountryPool(country=c, target_idx=idx)
        for f in fields:
            cpool.fields[f] = build_field_index(table[f].take(pa.array(idx)), FIELD_MODES[f])
            if log:
                fi = cpool.fields[f]
                log(f"{split}/{c}/{f}: {len(idx):,} targets, {fi.n_tokens:,} tokens, "
                    f"{fi.inverted.nnz:,} postings, {fi.nbytes() / 2**20:,.0f} MB")
        pool.countries[c] = cpool
    return pool


def df_summary(index, top=20):
    """DF distribution of one ``FieldIndex``: quantiles, tokens and postings above thresholds."""
    df = index.df
    total_postings = int(df.sum())
    out = {
        "n_docs": index.n_docs,
        "n_tokens": int(len(df)),
        "postings": total_postings,
        "df_quantiles": {q: int(np.quantile(df, q / 100)) for q in (50, 90, 99, 99.9)} if len(df) else {},
        "df_max": int(df.max()) if len(df) else 0,
    }
    for cap in (100, 1000, 10000, 100000):
        above = df > cap
        out[f"tokens_df_gt_{cap}"] = int(above.sum())
        out[f"postings_share_df_gt_{cap}"] = round(float(df[above].sum() / max(total_postings, 1)), 4)
    order = np.argsort(-df, kind="stable")[:top]
    out["top_tokens"] = [[index.vocab[int(i)].as_py(), int(df[i])] for i in order]
    return out
