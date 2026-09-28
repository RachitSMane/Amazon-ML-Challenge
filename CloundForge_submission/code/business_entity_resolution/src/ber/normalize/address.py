"""Deterministic address normalization and number extraction (no geographic lookup).

``normalize_address`` returns an ``Address`` with:

* ``ws``: the raw address with whitespace collapsed (case, punctuation and placeholders kept).
* ``missing``: True when nothing real is left after removing placeholder components.
  Addresses are comma-separated components and ``<NULL>`` / ``NULL`` / ``null`` occur as
  **components** (e.g. ``2612 SOUTHVIEW COURT, <NULL>, BRAINERD, MN``), so placeholders are
  removed per component and the rest of the address is kept.
* ``punct``: NFKC, casefold, placeholders removed, Latin accents folded, punctuation turned
  into spaces (the same rules as names, see ``text.strip_punctuation``).
* ``norm``: ``punct`` tokens with numbers canonicalized (``06252`` -> ``6252``, since sources
  zero-pad house numbers differently) and abbreviations expanded (``rd`` -> ``road``).
  Space-separated, so ``norm.split()`` is the token representation.
* ``script``: script class of the address (see ``text.script_of``).
* ``postcode``: the last country-shaped postcode that is not the first token of the address
  (US ``12345[-6789]`` ending a component, India ``123456``/``123 456``, France ``12345``,
  other countries a 4-6 digit number ending a component), digits only; "" if none.
* ``house_numbers``: numbers after a house marker (``no``, ``h no``, ``door no``, ``plot [no]``,
  ``n°``) plus a number that starts the address (``1948 weaver ...``, ``##8 willow ...``,
  ``g-58/1, ...``); France joins ``bis``/``ter``/``quater`` (``17 bis`` -> ``17bis``).
* ``unit_numbers``: numbers after a unit marker (suite, ste [not France], unit, apt, flat,
  shop, office, room, floor, building, bureau, bat ...).
* ``numbers``: every all-digit token of ``norm`` except the postcode.

Number values are space-separated, deduplicated, in order of appearance, with ``#`` and
spaces removed, ``/`` written as ``-`` and leading zeros dropped from every digit run.
``#`` is treated as noise, not as a unit marker: the data uses it before house numbers in
every country (``##8 Willow Oak Lane``, ``No #51``, ``Plot A-##26``).
"""

import re
from typing import NamedTuple

from .text import fold, is_placeholder, normalize_whitespace, script_of, strip_punctuation, unicode_base

# --------------------------------------------------------------------------- abbreviations

# token -> expansion, applied to whole tokens of the punctuation-free address. Explicit and
# per country because the same token differs: "st" is "street" in the US and "saint" in
# France, "dr" is "drive" in the US but usually "doctor" in Indian street names.
GLOBAL_ABBREVIATIONS = {"rd": "road", "ave": "avenue", "blvd": "boulevard"}

ABBREVIATIONS = {
    "US": {
        "st": "street", "str": "street", "av": "avenue", "dr": "drive", "ln": "lane", "ct": "court",
        "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway", "fwy": "freeway",
        "expy": "expressway", "ter": "terrace", "terr": "terrace", "trl": "trail", "sq": "square",
        "ctr": "center", "hts": "heights", "jct": "junction", "twp": "township", "bldg": "building",
        "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "rm": "room",
        # Single-letter compass points only: "ne" is also Nebraska ("<NULL>, NE, OMAHA").
        "n": "north", "s": "south", "e": "east", "w": "west",
    },
    "India": {
        "st": "street", "nr": "near", "opp": "opposite", "bldg": "building", "apt": "apartment",
        "apts": "apartments", "fl": "floor", "flr": "floor", "blk": "block", "sec": "sector",
        "sect": "sector", "ph": "phase", "extn": "extension", "ext": "extension", "mkt": "market",
        "clny": "colony", "col": "colony", "soc": "society", "hsg": "housing", "ind": "industrial",
        "indl": "industrial", "estt": "estate", "est": "estate", "dist": "district",
        "distt": "district", "vill": "village", "vil": "village", "stn": "station", "ngr": "nagar",
    },
    "France": {
        "r": "rue", "av": "avenue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
        "bvd": "boulevard", "pl": "place", "ch": "chemin", "chem": "chemin", "rte": "route",
        "imp": "impasse", "all": "allee", "sq": "square", "fg": "faubourg", "fbg": "faubourg",
        "crs": "cours", "qu": "quai", "qua": "quai", "pass": "passage", "res": "residence",
        "prom": "promenade", "st": "saint", "ste": "sainte", "bat": "batiment", "appt": "appartement",
    },
}

# --------------------------------------------------------------------------- number patterns

# A house/unit value: optional letter and hyphen, digits, optional letter, then more
# "/"- or "-"-separated parts: 1948, 221b, g-58/1, 3829/3250, 166-a, a11.
_VALUE = r"#*\s*([a-z]?-?#*\d+[a-z]?(?:\s*[/-]\s*#*[a-z]?\d+[a-z]?)*)(?![a-z0-9])"
_SEP = r"\s*\.?\s*[.:#-]*\s*"

_HOUSE_MARKER_RE = re.compile(
    r"\b(?:(?:h|d|door|house|plot|pt|site)\s*\.?\s*(?:no|nos|number)\b|plot|no|nos|number|num)" + _SEP + _VALUE
)
_LEADING_RE = re.compile(r"^[#\s]*" + _VALUE)
_UNIT_WORDS = ("suite", "unit", "apt", "apartment", "appt", "flat", "shop", "office", "room", "rm",
               "bureau", "bat", "batiment", "bldg", "building", "fl", "flr", "floor")
_UNIT_RE = {
    france: re.compile(
        r"\b(?:" + "|".join(_UNIT_WORDS + (() if france else ("ste",))) + r")\b"
        + r"\s*\.?\s*(?:no|nos|number)?" + _SEP + r"(?:" + _VALUE + r"|([a-z])\b)"
    )
    for france in (False, True)
}
_FR_SUFFIX_RE = re.compile(r"^[#\s]*(\d+)\s+(bis|ter|quater)\b")
_NUMERO_RE = re.compile(r"\bn\s*[°˚]\s*|\bno(?=\d)")
_DIGITS_RE = re.compile(r"\d+")

_POSTCODE_RE = {
    "US": re.compile(r"\b(\d{5})(?:-\d{4})?(?=\s*(?:,|$))"),
    "India": re.compile(r"\b(\d{3})\s?(\d{3})\b"),
    "France": re.compile(r"\b(\d{5})\b"),
}
_GENERIC_POSTCODE_RE = re.compile(r"\b(\d{4,6})(?=\s*(?:,|$))")


class Address(NamedTuple):
    ws: str
    missing: bool
    punct: str
    norm: str
    script: str
    postcode: str
    house_numbers: str
    unit_numbers: str
    numbers: str


def _canonical_number(value):
    """``#`` and spaces removed, ``/`` -> ``-``, leading zeros dropped from every digit run."""
    value = value.replace("#", "").replace(" ", "").replace("/", "-")
    return _DIGITS_RE.sub(lambda m: m.group(0).lstrip("0") or "0", value)


def _unique(values):
    seen = []
    for v in values:
        if v and v not in seen:
            seen.append(v)
    return seen


def remove_placeholder_components(text):
    """Drop comma-separated components that are empty or placeholders such as ``<NULL>``."""
    parts = [part.strip() for part in text.split(",")]
    return ", ".join(part for part in parts if part and not is_placeholder(part))


def extract_postcode(body, country):
    """Last postcode-shaped number that is not the first token of ``body`` ("" if none)."""
    pattern = _POSTCODE_RE.get(country, _GENERIC_POSTCODE_RE)
    found = ""
    for m in pattern.finditer(body):
        if re.search(r"[^\W_]", body[: m.start()]):
            found = "".join(g for g in m.groups() if g)
    return found


def extract_house_numbers(body, country, postcode=""):
    """House/building numbers (see the module docstring), excluding the canonical ``postcode``.

    A marker that belongs to a unit (``shop no 188``, ``flat no 4``) is left to the units.
    """
    unit_spans = [m.span() for m in _UNIT_RE[country == "France"].finditer(body)]
    values = []
    lead = _FR_SUFFIX_RE.match(body) if country == "France" else None
    if lead:
        values.append(lead.group(1) + lead.group(2))
    else:
        m = _LEADING_RE.match(body)
        if m:
            values.append(m.group(1))
    values.extend(
        m.group(1) for m in _HOUSE_MARKER_RE.finditer(body)
        if not any(start <= m.start() < end for start, end in unit_spans)
    )
    values = [_canonical_number(v) for v in values]
    return _unique(v for v in values if v != postcode)


def extract_unit_numbers(body, country):
    """Unit/suite/floor/shop numbers (see the module docstring)."""
    pattern = _UNIT_RE[country == "France"]
    return _unique(_canonical_number(m.group(1) or m.group(2)) for m in pattern.finditer(body))


def expand_tokens(tokens, country):
    """Canonicalize all-digit tokens and expand abbreviations for ``country``."""
    table = ABBREVIATIONS.get(country, {})
    out = []
    for token in tokens:
        if token.isdigit() and token.isascii():
            out.append(token.lstrip("0") or "0")
        else:
            out.append(table.get(token) or GLOBAL_ABBREVIATIONS.get(token, token))
    return out


_MISSING = dict(missing=True, punct="", norm="", script="empty", postcode="", house_numbers="",
                unit_numbers="", numbers="")


def normalize_address(raw, country=""):
    """Normalize one address; returns an ``Address`` (see the module docstring)."""
    raw = raw or ""
    ws = normalize_whitespace(raw)
    body = remove_placeholder_components(unicode_base(raw))
    if not re.search(r"[^\W_]", body):
        return Address(ws=ws, **_MISSING)

    body = _NUMERO_RE.sub("no ", fold(body))
    postcode = extract_postcode(body, country)
    canonical_postcode = _canonical_number(postcode) if postcode else ""
    house = extract_house_numbers(body, country, canonical_postcode)
    units = extract_unit_numbers(body, country)

    punct = strip_punctuation(body, country)
    tokens = expand_tokens(punct.split(), country)
    norm = " ".join(tokens)
    numbers = _unique(t for t in tokens if t.isdigit() and t != canonical_postcode)
    return Address(
        ws=ws,
        missing=False,
        punct=punct,
        norm=norm,
        script=script_of(body),
        postcode=postcode,
        house_numbers=" ".join(house),
        unit_numbers=" ".join(units),
        numbers=" ".join(numbers),
    )
