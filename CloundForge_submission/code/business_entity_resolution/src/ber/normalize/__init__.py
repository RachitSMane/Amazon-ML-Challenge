"""Normalization of business names and addresses into matching-ready representations.

* ``text``: Unicode/case/punctuation normalization, Latin accent folding, script detection.
* ``legal_forms``: explicit legal-form lists (global, US, India, France, Indic scripts).
* ``address``: placeholder removal, postcode/house/unit extraction, abbreviation expansion.
* ``cache``: the versioned ``work/cache/norm_v1`` Parquet cache built from the raw cache.

``normalize_table`` turns one Arrow table of raw records into the normalized schema
``NORMALIZED_SCHEMA``. It works one chunk at a time, so only that chunk's strings are ever
Python objects. No transliteration of Indic text happens here (a later phase).
"""

from functools import lru_cache
from typing import NamedTuple

import pyarrow as pa

from .address import normalize_address
from .legal_forms import strip_legal_forms
from .text import clean, fold, script_of

from ..config import NORM_CACHE_VERSION as NORMALIZATION_VERSION  # noqa: E402,F401  (bump in config)

NORMALIZED_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("source", pa.int8()),
    ("name_raw", pa.string()),
    ("name_norm", pa.string()),       # clean(): NFKC, casefold, punctuation -> space; accents kept
    ("name_fold", pa.string()),       # name_norm without Latin diacritics
    ("name_stripped", pa.string()),   # name_fold without legal forms
    ("name_key", pa.string()),        # sorted unique tokens of name_stripped
    ("legal_forms", pa.string()),     # canonical legal-form labels, space-separated
    ("name_script", pa.string()),
    ("address_raw", pa.string()),
    ("address_missing", pa.bool_()),
    ("address_ws", pa.string()),      # raw, whitespace collapsed
    ("address_punct", pa.string()),   # placeholders removed, cleaned and folded
    ("address_norm", pa.string()),    # address_punct with numbers and abbreviations canonical
    ("address_script", pa.string()),
    ("postcode", pa.string()),
    ("house_numbers", pa.string()),
    ("unit_numbers", pa.string()),
    ("address_numbers", pa.string()),
])


class Name(NamedTuple):
    norm: str
    fold: str
    stripped: str
    key: str
    legal_forms: str
    script: str


@lru_cache(maxsize=1 << 18)
def normalize_name(raw, country=""):
    """Normalize one business name; returns a ``Name``."""
    norm = clean(raw, country)
    folded = fold(norm)
    stripped, forms = strip_legal_forms(folded, country)
    key = " ".join(sorted(set(stripped.split())))
    return Name(norm, folded, stripped, key, " ".join(forms), script_of(norm))


@lru_cache(maxsize=1 << 18)
def _address(raw, country):
    return normalize_address(raw, country)


def normalize_table(table):
    """Normalize an Arrow table with the raw source columns; returns ``NORMALIZED_SCHEMA``.

    Input columns: entity_id, business_name, business_address, country, source. The ID,
    country, source and both raw strings are copied through unchanged.
    """
    names = table["business_name"].to_pylist()
    addresses = table["business_address"].to_pylist()
    countries = table["country"].to_pylist()

    name_cols = [[] for _ in Name._fields]
    addr_cols = [[] for _ in range(9)]
    for name, address, country in zip(names, addresses, countries):
        for col, value in zip(name_cols, normalize_name(name, country)):
            col.append(value)
        for col, value in zip(addr_cols, _address(address, country)):
            col.append(value)
    n_norm, n_fold, n_stripped, n_key, n_forms, n_script = name_cols
    a_ws, a_missing, a_punct, a_norm, a_script, a_post, a_house, a_unit, a_numbers = addr_cols

    computed = {
        "name_norm": n_norm, "name_fold": n_fold, "name_stripped": n_stripped, "name_key": n_key,
        "legal_forms": n_forms, "name_script": n_script, "address_missing": a_missing,
        "address_ws": a_ws, "address_punct": a_punct, "address_norm": a_norm,
        "address_script": a_script, "postcode": a_post, "house_numbers": a_house,
        "unit_numbers": a_unit, "address_numbers": a_numbers,
    }
    passthrough = {
        "entity_id": table["entity_id"], "country": table["country"], "source": table["source"],
        "name_raw": table["business_name"], "address_raw": table["business_address"],
    }
    columns = [
        passthrough[f.name] if f.name in passthrough else pa.array(computed[f.name], f.type)
        for f in NORMALIZED_SCHEMA
    ]
    return pa.Table.from_arrays(columns, schema=NORMALIZED_SCHEMA)
