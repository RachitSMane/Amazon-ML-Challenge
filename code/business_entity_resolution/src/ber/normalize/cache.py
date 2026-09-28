"""Versioned cache of normalized records: ``work/cache/norm_v{N}/{split}_source{n}.parquet``.

Built from the raw Parquet cache (``ber.io.build_raw_cache``), never from the TSVs, and
never touching either. Each source file is read in batches of ``batch_rows`` rows; the
batches are normalized by a pool of worker processes (at most ``2 * workers`` batches in
flight, so memory stays bounded) and written in their original order. Row order, entity
IDs, countries and sources are therefore identical to the raw file.

A cache entry is fresh only while all of these still hold (see ``is_norm_fresh``):

* the raw-cache entry itself is fresh (the raw TSV is unchanged);
* the raw-cache Parquet file has the size and mtime recorded at build time;
* the normalization code (every ``.py`` file of this package) has the recorded SHA-256;
* the manifest version equals ``config.NORM_CACHE_VERSION``.

The manifest format and atomic writes are shared with the raw cache (``ber.io``).

CLI (from ``code/business_entity_resolution/src``)::

    python -m ber.normalize.cache build [--workers N] [--force] [--data-dir DIR] [--work-dir DIR]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .. import io
from ..config import NORM_CACHE_VERSION, SOURCES, SPLITS, Config
from . import NORMALIZED_SCHEMA, normalize_table

PARQUET_COMPRESSION = "zstd"
RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country", "source"]
CATEGORY_COLUMNS = ("country", "name_script", "address_script")


def source_keys():
    """The six source keys (``train_source1`` ... ``test_source3``)."""
    return [io.source_key(split, source) for split in SPLITS for source in SOURCES]


def code_fingerprint():
    """SHA-256 over the source of every module in ``ber.normalize`` (sorted by name)."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def norm_file(cfg, key):
    return cfg.norm_cache_dir / f"{key}.parquet"


def read_norm_manifest(cfg):
    return io.load_manifest(cfg.norm_cache_dir, NORM_CACHE_VERSION)


def is_norm_fresh(cfg, key, manifest=None, fingerprint=None):
    """True if the normalized file of ``key`` exists and all its inputs are unchanged."""
    entry = (manifest or read_norm_manifest(cfg))["entries"].get(key)
    if not entry or not norm_file(cfg, key).is_file():
        return False
    if not io.is_cache_fresh(cfg, key):
        return False
    if entry.get("raw_cache_file") != io.file_stamp(io.raw_cache_file(cfg, key)):
        return False
    return entry.get("code_sha256") == (fingerprint or code_fingerprint())


# --------------------------------------------------------------------------- building

def _normalize_task(task):
    """Worker entry point: normalize one raw batch. Top-level so it pickles under spawn."""
    key, table = task
    return key, normalize_table(table)


def _tasks(cfg, keys, batch_rows):
    for key in keys:
        parquet = pq.ParquetFile(io.raw_cache_file(cfg, key))
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=RAW_COLUMNS):
            yield key, pa.Table.from_batches([batch])


class _Writer:
    """Writes the normalized batches of one key, then finalizes the file and manifest entry."""

    def __init__(self, cfg, key, manifest, fingerprint):
        self.cfg, self.key, self.manifest, self.fingerprint = cfg, key, manifest, fingerprint
        self.path = norm_file(cfg, key)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.writer = pq.ParquetWriter(self.tmp, NORMALIZED_SCHEMA, compression=PARQUET_COMPRESSION)
        self.rows = 0
        self.start = time.time()

    def write(self, table):
        self.writer.write_table(table)
        self.rows += len(table)

    def close(self, raw_stamp_before, log):
        self.writer.close()
        expected = io.read_manifest(self.cfg)["entries"][self.key]["rows"]
        if self.rows != expected:
            os.remove(self.tmp)
            raise RuntimeError(f"{self.key}: wrote {self.rows} rows, raw cache has {expected}")
        if io.file_stamp(io.raw_cache_file(self.cfg, self.key)) != raw_stamp_before:
            os.remove(self.tmp)
            raise RuntimeError(f"{self.key}: raw cache changed during normalization; rebuild")
        os.replace(self.tmp, self.path)
        self.manifest["entries"][self.key] = {
            "rows": self.rows,
            "bytes": self.path.stat().st_size,
            "raw_cache_file": raw_stamp_before,
            "code_sha256": self.fingerprint,
            "seconds": round(time.time() - self.start, 1),
        }
        io.save_manifest(self.cfg.norm_cache_dir, self.manifest)
        if log:
            log(f"{self.key}: {self.rows:,} rows in {time.time() - self.start:.1f}s")


def build_norm_cache(cfg, force=False, workers=None, batch_rows=100_000, keys=None, log=None):
    """Create or refresh the normalized cache for ``keys`` (default: all six sources).

    ``workers``: processes to use (default: CPU count - 1; 1 runs in this process).
    Requires a fresh raw cache for every key. Returns the manifest.
    """
    cfg.validate()
    keys = list(keys or source_keys())
    stale_raw = [k for k in keys if not io.is_cache_fresh(cfg, k)]
    if stale_raw:
        raise RuntimeError(f"raw cache missing or stale for {stale_raw}; run `python -m ber.io build-cache`")
    cfg.norm_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_norm_manifest(cfg)
    fingerprint = code_fingerprint()
    todo = [k for k in keys if force or not is_norm_fresh(cfg, k, manifest, fingerprint)]
    for key in keys:
        if key not in todo and log:
            log(f"{key}: fresh, skipped")
    if not todo:
        return manifest

    stamps = {k: io.file_stamp(io.raw_cache_file(cfg, k)) for k in todo}
    workers = max(1, workers or (os.cpu_count() or 2) - 1)
    writers = {}

    def handle(result):
        key, table = result
        if key not in writers:
            for other in list(writers):  # keys arrive in order: the previous one is complete
                writers.pop(other).close(stamps[other], log)
            writers[key] = _Writer(cfg, key, manifest, fingerprint)
        writers[key].write(table)

    try:
        if workers == 1:
            for task in _tasks(cfg, todo, batch_rows):
                handle(_normalize_task(task))
        else:
            with get_context("spawn").Pool(workers) as pool:
                pending = deque()
                for task in _tasks(cfg, todo, batch_rows):
                    pending.append(pool.apply_async(_normalize_task, (task,)))
                    if len(pending) >= 2 * workers:
                        handle(pending.popleft().get())
                while pending:
                    handle(pending.popleft().get())
        for key in list(writers):
            writers.pop(key).close(stamps[key], log)
    finally:
        for w in writers.values():  # only on error: drop partial files
            w.writer.close()
            if w.tmp.exists():
                os.remove(w.tmp)
    return manifest


# --------------------------------------------------------------------------- loading

def read_normalized_table(cfg, split, source, columns=None):
    """Arrow table of normalized records; raises if the cache entry is missing or stale."""
    key = io.source_key(split, source)
    if not is_norm_fresh(cfg, key):
        raise RuntimeError(f"normalized cache missing or stale for {key}; run `python -m ber.normalize.cache build`")
    return pq.read_table(norm_file(cfg, key), columns=columns)


def to_frame(table):
    """pandas view: strings stay in Arrow; country and script columns become categories."""
    for name in CATEGORY_COLUMNS:
        if name in table.column_names:
            i = table.column_names.index(name)
            table = table.set_column(i, name, table[name].dictionary_encode())
    return table.to_pandas(types_mapper=io._string_dtype)


def load_normalized(cfg, split, source, columns=None):
    """Load ``{split}_source{source}`` from the normalized cache as a DataFrame."""
    return to_frame(read_normalized_table(cfg, split, source, columns))


def iter_normalized_chunks(cfg, split, source, batch_rows=500_000, columns=None):
    """Yield the normalized records of one source as DataFrame chunks of ``batch_rows`` rows."""
    key = io.source_key(split, source)
    if not is_norm_fresh(cfg, key):
        raise RuntimeError(f"normalized cache missing or stale for {key}")
    for batch in pq.ParquetFile(norm_file(cfg, key)).iter_batches(batch_size=batch_rows, columns=columns):
        yield to_frame(pa.Table.from_batches([batch]))


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the normalized Parquet cache.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create or refresh the normalized cache")
    build.add_argument("--data-dir", help="dataset directory (default: auto-discovered)")
    build.add_argument("--work-dir", help="work directory (default: $BER_WORK_DIR or <repo>/work)")
    build.add_argument("--workers", type=int, help="worker processes (default: CPU count - 1)")
    build.add_argument("--force", action="store_true", help="rebuild even fresh entries")
    args = parser.parse_args(argv)

    cfg = Config.discover(args.data_dir, args.work_dir)
    print(f"raw cache:  {cfg.raw_cache_dir}\nnorm cache: {cfg.norm_cache_dir}", file=sys.stderr)
    manifest = build_norm_cache(cfg, force=args.force, workers=args.workers,
                                log=lambda msg: print(msg, file=sys.stderr))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
