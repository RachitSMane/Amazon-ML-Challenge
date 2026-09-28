"""Loaders for the source files and the training ground truth, with a Parquet cache.

TSV parsing rules (they match the official validator, which splits each line on tabs):

* The delimiter is a tab and **quote characters are literal**. Some names and addresses
  start with ``"`` (e.g. 134 rows in test_source1), and a quote-aware reader would change
  them.
* Every value is read as a string. An empty field stays ``""`` (never null), so empty
  S2/S3 addresses are kept exactly as they appear in the file.
* A row with the wrong number of fields, or a header that differs from the expected
  columns, raises instead of being skipped.

Parquet cache (``build_raw_cache``): each raw TSV is streamed block by block into a
zstd-compressed Parquet file under ``cfg.raw_cache_dir``, so at most one block is in memory
while writing. The raw TSVs are only ever opened for reading. ``manifest.json`` records the
size and modification time of each raw file; a cache entry is used only while they still
match, otherwise the loaders read the TSV directly. The cache holds exactly the parsed
values (no normalization) and can be deleted and rebuilt at any time.

In-memory representation: text columns stay in Arrow buffers (pandas ``string[pyarrow]``),
so the ~10M target records never become Python objects. ``country`` is dictionary-encoded
(a pandas category) and ``source`` is an ``int8`` (1, 2 or 3), checked against every ID
prefix.

CLI (from ``code/business_entity_resolution/src``)::

    python -m ber.io build-cache [--data-dir DIR] [--work-dir DIR] [--force]
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .config import GROUND_TRUTH_COLUMNS, RAW_CACHE_VERSION, SOURCE_COLUMNS, SOURCES, SPLITS, Config

MANIFEST = "manifest.json"
GROUND_TRUTH_KEY = "train_ground_truth"
PARQUET_COMPRESSION = "zstd"


# --------------------------------------------------------------------------- TSV reading

def _parse_options():
    """Tab-separated, no quoting or escaping, no embedded newlines."""
    return pacsv.ParseOptions(
        delimiter="\t",
        quote_char=False,
        double_quote=False,
        escape_char=False,
        newlines_in_values=False,
        ignore_empty_lines=True,
    )


def _convert_options(columns):
    """Read every column as a non-null string."""
    return pacsv.ConvertOptions(
        column_types={name: pa.string() for name in columns},
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )


def _read_options(block_size):
    return pacsv.ReadOptions(block_size=block_size, use_threads=True, encoding="utf8")


def _check_header(names, expected, path):
    if tuple(names) != tuple(expected):
        raise ValueError(f"{path}: header {list(names)} != expected {list(expected)}")


def _finish_source(table, source, path):
    """Validate the ID prefixes of one source table and append the int8 ``source`` column."""
    prefix = f"S{source}-"
    bad = len(table) - pc.sum(pc.starts_with(table["entity_id"], prefix)).as_py() if len(table) else 0
    if bad:
        raise ValueError(f"{path}: {bad} entity_id values do not start with {prefix!r}")
    return table.append_column("source", pa.array(np.full(len(table), source, dtype=np.int8)))


def read_source_tsv(path, source, block_size=64 << 20):
    """Read one raw ``*_source{1,2,3}.tsv`` file into an Arrow table.

    Columns: entity_id, business_name, business_address, country (all string) and source (int8).
    """
    table = pacsv.read_csv(
        path,
        read_options=_read_options(block_size),
        parse_options=_parse_options(),
        convert_options=_convert_options(SOURCE_COLUMNS),
    )
    _check_header(table.column_names, SOURCE_COLUMNS, path)
    return _finish_source(table, source, path)


def iter_source_tsv(path, source, block_size=64 << 20):
    """Stream one raw source file as Arrow tables of roughly ``block_size`` bytes each."""
    reader = pacsv.open_csv(
        path,
        read_options=_read_options(block_size),
        parse_options=_parse_options(),
        convert_options=_convert_options(SOURCE_COLUMNS),
    )
    _check_header(reader.schema.names, SOURCE_COLUMNS, path)
    for batch in reader:
        yield _finish_source(pa.Table.from_batches([batch]), source, path)


def read_ground_truth_tsv(path, block_size=64 << 20):
    """Parse the raw ``train_ground_truth.tsv`` into ``(s1, pairs)`` Arrow tables.

    ``s1``: source1_entity_id, n_matches (int16). ``pairs``: source1_entity_id,
    target_entity_id, target_source (int8). The match list is comma-separated; blanks around
    IDs and empty items are ignored (the same tolerance as ``evaluation.parse_id_list``).
    Raises ``ValueError`` on a bad prefix.
    """
    table = pacsv.read_csv(
        path,
        read_options=_read_options(block_size),
        parse_options=_parse_options(),
        convert_options=_convert_options(GROUND_TRUTH_COLUMNS),
    )
    _check_header(table.column_names, GROUND_TRUTH_COLUMNS, path)
    s1_ids = table["source1_entity_id"].combine_chunks()
    bad = len(s1_ids) - pc.sum(pc.starts_with(s1_ids, "S1-")).as_py() if len(s1_ids) else 0
    if bad:
        raise ValueError(f"{path}: {bad} source1_entity_id values do not start with 'S1-'")

    lists = pc.split_pattern(table["matched_entity_ids"].combine_chunks(), ",")
    parents = pc.list_parent_indices(lists).to_numpy()
    targets = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    keep = pc.not_equal(targets, "")
    targets = targets.filter(keep)
    parents = parents[keep.to_numpy(zero_copy_only=False)]

    prefix = pc.utf8_slice_codeunits(targets, 0, 3)
    is_s2 = pc.equal(prefix, "S2-").to_numpy(zero_copy_only=False)
    is_s3 = pc.equal(prefix, "S3-").to_numpy(zero_copy_only=False)
    bad = int((~(is_s2 | is_s3)).sum())
    if bad:
        raise ValueError(f"{path}: {bad} matched IDs do not start with 'S2-' or 'S3-'")

    n_matches = np.bincount(parents, minlength=len(s1_ids)).astype(np.int16)
    s1 = pa.table({"source1_entity_id": s1_ids, "n_matches": n_matches})
    pairs = pa.table({
        "source1_entity_id": s1_ids.take(pa.array(parents)),
        "target_entity_id": targets,
        "target_source": pa.array(np.where(is_s2, 2, 3).astype(np.int8)),
    })
    return s1, pairs


# --------------------------------------------------------------------------- Parquet cache

def source_key(split, source):
    """Cache key (and Parquet file stem) of one source file, e.g. ``train_source2``."""
    return f"{split}_source{source}"


def _cache_files(key):
    if key == GROUND_TRUTH_KEY:
        return [f"{key}_s1.parquet", f"{key}_pairs.parquet"]
    return [f"{key}.parquet"]


def _raw_path(cfg, key):
    if key == GROUND_TRUTH_KEY:
        return cfg.ground_truth_path
    split, source = key.split("_source")
    return cfg.source_path(split, int(source))


def file_stamp(path):
    """``{"bytes": size, "mtime_ns": mtime}`` of a file, used to detect changes cheaply."""
    st = os.stat(path)
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _raw_stamp(path):
    stamp = file_stamp(path)
    return {"raw_bytes": stamp["bytes"], "raw_mtime_ns": stamp["mtime_ns"]}


def load_manifest(directory, version):
    """Manifest of a cache folder (``{"version": .., "entries": {key: {...}}}``).

    Returns an empty manifest if the file is missing or was written by another version.
    Shared by every versioned cache (raw here, normalized in ``ber.normalize.cache``).
    """
    path = Path(directory) / MANIFEST
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("version") == version:
            return manifest
    return {"version": version, "entries": {}}


def save_manifest(directory, manifest):
    """Write a manifest atomically (temporary file, then rename)."""
    path = Path(directory) / MANIFEST
    tmp = path.with_name(MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_manifest(cfg):
    """Manifest of the raw Parquet cache."""
    return load_manifest(cfg.raw_cache_dir, RAW_CACHE_VERSION)


def _write_manifest(cfg, manifest):
    save_manifest(cfg.raw_cache_dir, manifest)


def raw_cache_file(cfg, key):
    """Path of the raw-cache Parquet file of a source key (e.g. ``train_source2``)."""
    return cfg.raw_cache_dir / _cache_files(key)[0]


def is_cache_fresh(cfg, key, manifest=None):
    """True if the cache entry exists and its raw file's size and mtime are unchanged."""
    entry = (manifest or read_manifest(cfg))["entries"].get(key)
    if not entry:
        return False
    if any(not (cfg.raw_cache_dir / name).is_file() for name in _cache_files(key)):
        return False
    stamp = _raw_stamp(_raw_path(cfg, key))
    return all(entry.get(k) == v for k, v in stamp.items())


def _atomic_parquet_path(path):
    return path.with_name(path.name + ".tmp")


def _write_source_cache(cfg, split, source):
    """Stream one raw source TSV into Parquet; return the number of rows written."""
    out = cfg.raw_cache_dir / _cache_files(source_key(split, source))[0]
    tmp = _atomic_parquet_path(out)
    rows, writer = 0, None
    try:
        for table in iter_source_tsv(cfg.source_path(split, source), source, cfg.block_size):
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=PARQUET_COMPRESSION)
            writer.write_table(table)
            rows += len(table)
        if writer is None:  # header-only file: still write a typed, empty table
            empty = _finish_source(pa.table({c: pa.array([], pa.string()) for c in SOURCE_COLUMNS}), source, out)
            pq.write_table(empty, tmp, compression=PARQUET_COMPRESSION)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, out)
    return rows


def _write_ground_truth_cache(cfg):
    s1, pairs = read_ground_truth_tsv(cfg.ground_truth_path, cfg.block_size)
    for table, name in zip((s1, pairs), _cache_files(GROUND_TRUTH_KEY)):
        out = cfg.raw_cache_dir / name
        tmp = _atomic_parquet_path(out)
        pq.write_table(table, tmp, compression=PARQUET_COMPRESSION)
        os.replace(tmp, out)
    return len(s1)


def cache_keys():
    """All cache keys: the six source files, then the ground truth."""
    return [source_key(split, source) for split in SPLITS for source in SOURCES] + [GROUND_TRUTH_KEY]


def build_raw_cache(cfg, force=False, log=None):
    """Create or refresh the Parquet cache of every raw file; return the manifest.

    Entries whose raw file is unchanged are skipped unless ``force``. Files are written to a
    temporary name and renamed, and the manifest is saved after each entry, so an
    interrupted build never leaves a half-written file that looks valid.
    """
    cfg.validate()
    cfg.raw_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(cfg)
    for key in cache_keys():
        if not force and is_cache_fresh(cfg, key, manifest):
            if log:
                log(f"{key}: fresh, skipped")
            continue
        raw = _raw_path(cfg, key)
        stamp_before = _raw_stamp(raw)
        start = time.time()
        if key == GROUND_TRUTH_KEY:
            rows = _write_ground_truth_cache(cfg)
        else:
            split, source = key.split("_source")
            rows = _write_source_cache(cfg, split, int(source))
        if _raw_stamp(raw) != stamp_before:
            raise RuntimeError(f"{raw} changed while it was being cached; rebuild")
        manifest["entries"][key] = {
            "raw": str(Path(raw).relative_to(cfg.data_dir).as_posix()),
            **stamp_before,
            "rows": rows,
            "files": {name: (cfg.raw_cache_dir / name).stat().st_size for name in _cache_files(key)},
        }
        _write_manifest(cfg, manifest)
        if log:
            log(f"{key}: {rows:,} rows in {time.time() - start:.1f}s")
    return manifest


# --------------------------------------------------------------------------- loaders

def read_source_table(cfg, split, source, use_cache=True):
    """Arrow table of one source file, from the cache if fresh, otherwise from the TSV."""
    key = source_key(split, source)
    if use_cache and is_cache_fresh(cfg, key):
        return pq.read_table(cfg.raw_cache_dir / _cache_files(key)[0])
    return read_source_tsv(cfg.source_path(split, source), source, cfg.block_size)


def _string_dtype(arrow_type):
    """Map Arrow strings to pandas ``string[pyarrow]`` (no Python objects)."""
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return pd.StringDtype("pyarrow")
    return None


def to_frame(table):
    """Convert a source table to pandas: strings stay in Arrow, ``country`` becomes a category."""
    if "country" in table.column_names:
        i = table.column_names.index("country")
        table = table.set_column(i, "country", pc.dictionary_encode(table["country"]))
    return table.to_pandas(types_mapper=_string_dtype)


def load_source(cfg, split, source, use_cache=True):
    """Load ``{split}_source{source}`` as a DataFrame (see ``to_frame``)."""
    return to_frame(read_source_table(cfg, split, source, use_cache))


def iter_source_chunks(cfg, split, source, block_size=None, use_cache=True):
    """Yield ``{split}_source{source}`` as DataFrame chunks of about ``block_size`` bytes.

    Country categories are encoded per chunk; use ``load_source`` for one consistent encoding.
    """
    block_size = block_size or cfg.block_size
    key = source_key(split, source)
    manifest = read_manifest(cfg)
    if use_cache and is_cache_fresh(cfg, key, manifest):
        entry = manifest["entries"][key]
        bytes_per_row = entry["raw_bytes"] / max(entry["rows"], 1)
        batch_rows = max(1, int(block_size / bytes_per_row))
        parquet = pq.ParquetFile(cfg.raw_cache_dir / _cache_files(key)[0])
        for batch in parquet.iter_batches(batch_size=batch_rows):
            yield to_frame(pa.Table.from_batches([batch]))
        return
    for table in iter_source_tsv(cfg.source_path(split, source), source, block_size):
        yield to_frame(table)


def load_targets(cfg, split, use_cache=True):
    """Load the S2 and S3 files of ``split`` as one DataFrame (S2 rows first, then S3).

    ``source`` tells the two apart; ``country`` shares one category encoding across both.
    """
    tables = [read_source_table(cfg, split, s, use_cache) for s in (2, 3)]
    return to_frame(pa.concat_tables(tables))


@dataclass
class GroundTruth:
    """The training ground truth in two shapes.

    ``s1``: one row per S1 entity, in file order, with ``source1_entity_id`` and ``n_matches``
    (int16; 0 for an entity with no match).
    ``pairs``: one row per true pair, with ``source1_entity_id``, ``target_entity_id`` and
    ``target_source`` (int8, 2 or 3).
    """

    s1: pd.DataFrame
    pairs: pd.DataFrame


def load_ground_truth(cfg, use_cache=True):
    """Load the training ground truth, from the cache if fresh, otherwise from the TSV."""
    if use_cache and is_cache_fresh(cfg, GROUND_TRUTH_KEY):
        s1, pairs = (pq.read_table(cfg.raw_cache_dir / name) for name in _cache_files(GROUND_TRUTH_KEY))
    else:
        s1, pairs = read_ground_truth_tsv(cfg.ground_truth_path, cfg.block_size)
    return GroundTruth(
        s1=s1.to_pandas(types_mapper=_string_dtype),
        pairs=pairs.to_pandas(types_mapper=_string_dtype),
    )


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the Parquet cache of the raw dataset files.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-cache", help="create or refresh the raw Parquet cache")
    build.add_argument("--data-dir", help="dataset directory (default: auto-discovered)")
    build.add_argument("--work-dir", help="work directory (default: $BER_WORK_DIR or <repo>/work)")
    build.add_argument("--force", action="store_true", help="rebuild even fresh entries")
    args = parser.parse_args(argv)

    cfg = Config.discover(args.data_dir, args.work_dir)
    print(f"data_dir:  {cfg.data_dir}\ncache_dir: {cfg.raw_cache_dir}", file=sys.stderr)
    manifest = build_raw_cache(cfg, force=args.force, log=lambda msg: print(msg, file=sys.stderr))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
