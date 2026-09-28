"""Deterministic text normalization and script detection for business names.

Stages of ``clean`` (each one is idempotent and never drops Indic letters or vowel signs):

1. Unicode NFKC (fullwidth forms, ligatures, ``º`` -> ``o``); skipped for pure-ASCII input.
2. ``casefold`` (``ß`` -> ``ss``).
3. Remove invisible characters: zero-width (non-)joiners, zero-width space, soft hyphen, BOM.
   Indic spellings of the same word differ only by ZWNJ in the data (e.g. Oriya/Telugu "LLP").
4. A value that is only a missing-value placeholder (``<NULL>``, ``null``, ...) becomes "".
5. Apostrophes: for France an elision (``l'``, ``d'``, ``qu'`` ...) becomes a space so that
   ``d'alzon`` gives the tokens ``d alzon``; everywhere else apostrophes are deleted so that
   ``McDonald's`` gives ``mcdonalds``.
6. ``&`` becomes ``et`` for France and ``and`` elsewhere (both sides of a true pair share the
   country, so either choice is consistent).
7. Dotted initialisms lose their dots: ``p.v.t.`` -> ``pvt``, ``u.s.a`` -> ``usa``.
8. Every other character that is not a letter, a digit or a combining mark becomes a space.
9. Whitespace is collapsed and trimmed.

``fold`` then removes Latin diacritics only (``é`` -> ``e``, ``œ`` -> ``oe``) with a fixed
translation table over the Latin blocks, so Indic combining marks are untouched.

Script classes (``script_of``): ``empty`` (no letters), ``latin``, one label per Indic script
(``devanagari``, ``bengali``, ``gurmukhi``, ``gujarati``, ``oriya``, ``tamil``, ``telugu``,
``kannada``, ``malayalam``), ``other`` (any other non-Latin script) and ``mixed`` (letters
from two or more of those classes). Digits and punctuation do not count as letters.
"""

import bisect
import re
import unicodedata

# Values that stand for "missing" (compared after casefold and trimming). Audited on the
# data: ``<NULL>`` and ``NULL``/``null`` occur as address components; names never contain them.
NULL_PLACEHOLDERS = frozenset({"<null>", "null", "none", "nan", "n/a"})

_INVISIBLE_RE = re.compile("[​‌‍⁠­﻿]")
_APOSTROPHES = "'’‘ʼ`´"
_APOSTROPHE_RE = re.compile(f"[{_APOSTROPHES}]")
# French elision: a clitic (l, d, j, m, n, s, t, c, qu, lorsqu, puisqu, jusqu) + apostrophe + letter.
_ELISION_RE = re.compile(rf"\b(l|d|j|m|n|s|t|c|qu|lorsqu|puisqu|jusqu)[{_APOSTROPHES}](?=\w)")
# Two or more "letter." groups, optionally followed by a final letter: p.v.t. / u.s.a / a.b.
_INITIALISM_RE = re.compile(r"\b(?:[^\W\d_]\.){2,}(?:[^\W\d_]\b)?")
# Keep letters/digits (\w) and combining marks: Latin diacritics (0300-036F) and the Indic
# blocks (0900-0DFF, which hold the vowel signs and viramas \w does not match).
_NON_WORD_RE = re.compile(r"[^\ẁ-ͯऀ-෿]|_")
_SPACE_RE = re.compile(r"\s+")

# --------------------------------------------------------------------------- scripts

INDIC_SCRIPTS = (
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
)
_LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))
SCRIPT_LABELS = ("empty", "latin") + tuple(label for _, _, label in INDIC_SCRIPTS) + ("other", "mixed")

# Sorted interval table for bisect: (start, end, label).
_SCRIPT_TABLE = sorted([(lo, hi, "latin") for lo, hi in _LATIN_RANGES] + list(INDIC_SCRIPTS))
_SCRIPT_STARTS = [row[0] for row in _SCRIPT_TABLE]


def _char_script(ch):
    """Script label of one letter (``other`` when it is outside the Latin and Indic blocks)."""
    cp = ord(ch)
    i = bisect.bisect_right(_SCRIPT_STARTS, cp) - 1
    if i >= 0 and cp <= _SCRIPT_TABLE[i][1]:
        return _SCRIPT_TABLE[i][2]
    return "other"


def script_of(text):
    """Classify ``text`` by the scripts of its letters (see the module docstring)."""
    if text.isascii():
        return "latin" if any(c.isalpha() for c in text) else "empty"
    found = set()
    for ch in text:
        # Indic vowel signs are marks, not letters, so they are classified with their block.
        if ch.isalpha() or (0x0900 <= ord(ch) <= 0x0DFF and unicodedata.category(ch)[0] == "M"):
            found.add(_char_script(ch))
            if len(found) > 1:
                return "mixed"
    return found.pop() if found else "empty"


# --------------------------------------------------------------------------- cleaning

def is_placeholder(text):
    """True if ``text`` is empty or only a missing-value placeholder such as ``<NULL>``."""
    return text.strip().casefold() in NULL_PLACEHOLDERS or not text.strip()


def _drop_initialism_dots(match):
    return match.group(0).replace(".", "")


def unicode_base(text):
    """Stages 1-3: NFKC, casefold and removal of invisible characters."""
    if not text.isascii():
        text = unicodedata.normalize("NFKC", text)
        text = _INVISIBLE_RE.sub("", text)
    return text.casefold()


def clean(text, country=""):
    """Normalize a name (stages 1-9). Accents are kept; see ``fold`` for accent removal."""
    if text is None:
        return ""
    text = unicode_base(text)
    if is_placeholder(text):
        return ""
    return strip_punctuation(text, country)


def strip_punctuation(text, country=""):
    """Stages 5-9 on text that already went through ``unicode_base``."""
    if country == "France":
        text = _ELISION_RE.sub(r"\1 ", text)
    text = _APOSTROPHE_RE.sub("", text)
    text = text.replace("&", " et " if country == "France" else " and ")
    if "." in text:
        text = _INITIALISM_RE.sub(_drop_initialism_dots, text)
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------- accent folding

# Letters that do not decompose under NFD but have a conventional ASCII spelling.
_SPECIAL_FOLDS = {"æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ð": "d", "ł": "l", "ı": "i", "þ": "th", "ŀ": "l"}


def _build_fold_table():
    table = {}
    for lo, hi in ((0x00C0, 0x024F), (0x1E00, 0x1EFF)):
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            base = "".join(c for c in unicodedata.normalize("NFD", ch) if not 0x0300 <= ord(c) <= 0x036F)
            if base != ch and base.isascii():
                table[cp] = base.casefold()
    for ch, repl in _SPECIAL_FOLDS.items():
        table[ord(ch)] = repl
        table[ord(ch.upper())] = repl
    for cp in range(0x0300, 0x0370):  # stray Latin combining diacritics
        table[cp] = None
    return table


_FOLD_TABLE = _build_fold_table()


def fold(text):
    """Remove Latin diacritics from already-cleaned text; other scripts are unchanged."""
    return text if text.isascii() else text.translate(_FOLD_TABLE)


def normalize_whitespace(text):
    """Collapse runs of whitespace to one space and trim (case and punctuation are kept)."""
    return _SPACE_RE.sub(" ", text).strip() if text else ""
