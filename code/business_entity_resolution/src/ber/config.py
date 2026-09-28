"""Paths, expected dataset layout and reference row counts.

Nothing here is machine-specific: every default is relative to the repository root, and
each location can be overridden with an argument or an environment variable.

Dataset directory (read-only, never copied into the repository). ``find_data_dir`` tries:

1. an explicit ``data_dir`` argument (e.g. from a CLI flag);
2. the ``BER_DATA_DIR`` environment variable;
3. ``<repo>/student_resource/dataset`` (the canonical location in the README);
4. ``<repo>/*_student_resource/student_resource/dataset`` (the official zip extracted in
   place, which is where the dataset lives on the development machine).

A candidate directory is accepted only if all seven expected TSV files exist in it.

Work directory (generated, gitignored, always safe to delete): an explicit ``work_dir``,
else ``BER_WORK_DIR``, else ``<repo>/work``. Caches live in versioned subfolders, so a
change of format never reads an old cache.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# src/ber/config.py -> ber -> src -> business_entity_resolution -> code -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR_ENV = "BER_DATA_DIR"
WORK_DIR_ENV = "BER_WORK_DIR"
DEFAULT_WORK_DIR = REPO_ROOT / "work"

# Bump when the raw Parquet cache format or the parsing rules change.
RAW_CACHE_VERSION = 1
# Bump when a normalization rule changes (ber.normalize); also invalidated by a code hash.
NORM_CACHE_VERSION = 1

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

# Relative paths inside the dataset directory.
SOURCE_FILES = {
    (split, source): f"{split}/{split}_source{source}.tsv" for split in SPLITS for source in SOURCES
}
GROUND_TRUTH_FILE = "train/train_ground_truth.tsv"
EXPECTED_FILES = tuple(SOURCE_FILES.values()) + (GROUND_TRUTH_FILE,)

# Data rows (header excluded), measured during the reconnaissance. Used as acceptance checks.
EXPECTED_ROWS = {
    ("train", 1): 2_206_821,
    ("train", 2): 5_034_616,
    ("train", 3): 5_285_603,
    ("test", 1): 1_732_544,
    ("test", 2): 4_887_273,
    ("test", 3): 5_082_316,
}
EXPECTED_GROUND_TRUTH_ROWS = 2_206_821
EXPECTED_GROUND_TRUTH_PAIRS = 7_638_365

# Accepted range for Config.block_size (bytes per Arrow read block).
MIN_BLOCK_SIZE = 1 << 20
MAX_BLOCK_SIZE = 1 << 30


def _candidate_dirs():
    """Return the implicit dataset directories to try, in priority order."""
    candidates = []
    if os.environ.get(DATA_DIR_ENV):
        candidates.append(Path(os.environ[DATA_DIR_ENV]))
    candidates.append(REPO_ROOT / "student_resource" / "dataset")
    candidates.extend(sorted(REPO_ROOT.glob("*_student_resource/student_resource/dataset")))
    return candidates


def missing_files(directory):
    """Return the expected dataset files that do not exist under ``directory``."""
    directory = Path(directory)
    return [name for name in EXPECTED_FILES if not (directory / name).is_file()]


def find_data_dir(data_dir=None):
    """Return the first candidate directory that contains every expected dataset file.

    An explicit ``data_dir`` that is incomplete raises immediately instead of silently
    falling back to another location. Raises ``FileNotFoundError`` listing what was tried.
    """
    if data_dir is not None:
        missing = missing_files(data_dir)
        if missing:
            raise FileNotFoundError(f"dataset directory {data_dir} is missing: {', '.join(missing)}")
        return Path(data_dir).resolve()
    tried = []
    for candidate in _candidate_dirs():
        if not missing_files(candidate):
            return candidate.resolve()
        tried.append(str(candidate))
    raise FileNotFoundError(
        f"no complete dataset found; set {DATA_DIR_ENV} or pass data_dir. Tried: {'; '.join(tried)}"
    )


def find_work_dir(work_dir=None):
    """Return the work directory (explicit > ``BER_WORK_DIR`` > ``<repo>/work``), resolved.

    The directory is not created here; writers create what they need.
    """
    if work_dir is None:
        work_dir = os.environ.get(WORK_DIR_ENV) or DEFAULT_WORK_DIR
    return Path(work_dir).resolve()


def _is_within(path, parent):
    return path == parent or parent in path.parents


@dataclass(frozen=True)
class Config:
    """Run-level settings shared by the pipeline stages."""

    data_dir: Path
    work_dir: Path = DEFAULT_WORK_DIR
    # Bytes per Arrow read block; also the approximate size of one chunk in chunked loading.
    block_size: int = 64 << 20

    @classmethod
    def discover(cls, data_dir=None, work_dir=None, **overrides):
        """Build and validate a Config with the dataset and work directories located."""
        cfg = cls(data_dir=find_data_dir(data_dir), work_dir=find_work_dir(work_dir), **overrides)
        cfg.validate()
        return cfg

    def validate(self):
        """Raise ``ValueError``/``FileNotFoundError`` if the configuration is unusable.

        Checks that the dataset is complete, that the work directory can never overlap the
        raw dataset (so generated files cannot touch it), and that ``block_size`` is sane.
        """
        missing = missing_files(self.data_dir)
        if missing:
            raise FileNotFoundError(f"dataset directory {self.data_dir} is missing: {', '.join(missing)}")
        data_dir, work_dir = Path(self.data_dir).resolve(), Path(self.work_dir).resolve()
        if _is_within(work_dir, data_dir) or _is_within(data_dir, work_dir):
            raise ValueError(f"work_dir {work_dir} must not overlap the dataset directory {data_dir}")
        if work_dir.exists() and not work_dir.is_dir():
            raise ValueError(f"work_dir {work_dir} exists and is not a directory")
        if not isinstance(self.block_size, int) or not MIN_BLOCK_SIZE <= self.block_size <= MAX_BLOCK_SIZE:
            raise ValueError(
                f"block_size must be an int in [{MIN_BLOCK_SIZE}, {MAX_BLOCK_SIZE}], got {self.block_size!r}"
            )

    def source_path(self, split, source):
        """Path of the raw ``{split}_source{source}.tsv``."""
        return Path(self.data_dir) / SOURCE_FILES[(split, source)]

    @property
    def ground_truth_path(self):
        """Path of the raw ``train_ground_truth.tsv``."""
        return Path(self.data_dir) / GROUND_TRUTH_FILE

    @property
    def raw_cache_dir(self):
        """Folder of the Parquet copies of the raw files (see ``ber.io.build_raw_cache``)."""
        return Path(self.work_dir) / "cache" / f"raw_v{RAW_CACHE_VERSION}"

    @property
    def norm_cache_dir(self):
        """Folder of the normalized records (see ``ber.normalize.cache``)."""
        return Path(self.work_dir) / "cache" / f"norm_v{NORM_CACHE_VERSION}"
