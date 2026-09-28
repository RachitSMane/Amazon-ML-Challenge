"""Experimental phonetic-skeleton channel for Latin S1 names vs Indic-script target names.

Measured in Phase 2: every dev S1 name is Latin, while 6.66% of true pairs have an
Indic-script target name that shares no name token with its S1. Most of those names are
English words written in an Indic script (``राम मार्केटिंग`` = "Ram Marketing"), so a coarse
phonetic key can bring both sides together without any external transliteration library.

1. ``romanize``: Indic characters -> Latin letters. The nine Brahmic blocks (Devanagari,
   Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) share the
   ISCII-derived layout, so each character is shifted into the Devanagari block and read
   from one table. Inherent vowels are not inserted (the skeleton drops vowels anyway).
2. ``skeleton_token``: a Latin word -> a consonant-class key. English spellings are first
   rewritten (``tion``/``sion`` -> ``shan``, ``tech`` -> ``tek``, soft ``c`` -> ``s``),
   digraphs are merged (``sh``->s, ``ph``->p, ``kh``->k ...), consonants are grouped into
   classes (k/g/q -> k, t/d -> t, p/b/f -> p, v/w -> v, s/z -> s, ch/j -> j), vowels,
   ``y`` and ``h`` are dropped except as the first letter (an initial vowel becomes ``a``),
   repeats are collapsed, and a final run of ``s``/``j`` (plural) is removed while the key
   has 4+ characters. Also: soft ``g`` (not word-initial, before e/i/y) -> ``j`` and
   ``tur`` -> ``chur`` (future, manufacturing). Known limitation: Tamil ``ச`` is read as
   ``ch`` although loanwords also use it for ``s``.
   Keys shorter than 2 characters are dropped.

Nothing here changes the Phase 1 normalized columns; the key is computed only for retrieval.
"""

import re
from functools import lru_cache

# --------------------------------------------------------------------------- romanization

_DEVANAGARI = {
    # consonants
    0x0915: "k", 0x0916: "kh", 0x0917: "g", 0x0918: "gh", 0x0919: "n", 0x091A: "ch", 0x091B: "chh",
    0x091C: "j", 0x091D: "jh", 0x091E: "n", 0x091F: "t", 0x0920: "th", 0x0921: "d", 0x0922: "dh",
    0x0923: "n", 0x0924: "t", 0x0925: "th", 0x0926: "d", 0x0927: "dh", 0x0928: "n", 0x0929: "n",
    0x092A: "p", 0x092B: "ph", 0x092C: "b", 0x092D: "bh", 0x092E: "m", 0x092F: "y", 0x0930: "r",
    0x0931: "r", 0x0932: "l", 0x0933: "l", 0x0934: "l", 0x0935: "v", 0x0936: "sh", 0x0937: "sh",
    0x0938: "s", 0x0939: "h",
    # nukta forms
    0x0958: "q", 0x0959: "kh", 0x095A: "g", 0x095B: "z", 0x095C: "r", 0x095D: "rh", 0x095E: "f", 0x095F: "y",
    # independent vowels
    0x0904: "a", 0x0905: "a", 0x0906: "aa", 0x0907: "i", 0x0908: "ii", 0x0909: "u", 0x090A: "uu",
    0x090B: "ri", 0x090C: "li", 0x090D: "e", 0x090E: "e", 0x090F: "e", 0x0910: "ai", 0x0911: "o",
    0x0912: "o", 0x0913: "o", 0x0914: "au", 0x0960: "ri", 0x0961: "li",
    # vowel signs
    0x093E: "aa", 0x093F: "i", 0x0940: "ii", 0x0941: "u", 0x0942: "uu", 0x0943: "ri", 0x0944: "ri",
    0x0945: "e", 0x0946: "e", 0x0947: "e", 0x0948: "ai", 0x0949: "o", 0x094A: "o", 0x094B: "o",
    0x094C: "au", 0x0962: "li", 0x0963: "li",
    # signs: candrabindu/anusvara -> n, visarga -> h, nukta/virama/avagraha -> nothing
    0x0900: "n", 0x0901: "n", 0x0902: "n", 0x0903: "h", 0x093C: "", 0x094D: "", 0x093D: "", 0x0950: "om",
    0x0964: " ", 0x0965: " ", 0x0970: "",
}
_DEVANAGARI.update({0x0966 + d: str(d) for d in range(10)})

# Characters without a Devanagari counterpart at the same offset.
_SPECIAL = {
    0x0A70: "n", 0x0A71: "", 0x0A72: "", 0x0A73: "",           # Gurmukhi tippi, addak, vowel bearers
    0x09CE: "t",                                             # Bengali khanda ta
    0x0B71: "v",                                             # Oriya wa
    0x0B83: "h",                                             # Tamil aytham
    0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",  # Malayalam chillus
    0x0D4E: "r",                                             # Malayalam dot reph
}
_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]


def _build_table():
    table = {}
    for base in _BLOCKS:
        for offset in range(0x80):
            cp = base + offset
            latin = _DEVANAGARI.get(0x0900 + offset)
            if latin is not None:
                table[cp] = latin
    table.update(_SPECIAL)
    for base in _BLOCKS:  # anything else in the Indic blocks is dropped
        for offset in range(0x80):
            table.setdefault(base + offset, "")
    return table


_ROMAN_TABLE = _build_table()


# Conjuncts whose sound differs from their letters (Malayalam "റ്റ" = tt, "ന്റ" = nt).
_CONJUNCTS = (("റ്റ", "t"), ("ന്റ", "nt"))


def romanize(text):
    """Map Indic characters to Latin letters; other characters are unchanged."""
    for a, b in _CONJUNCTS:
        if a in text:
            text = text.replace(a, b)
    return text.translate(_ROMAN_TABLE)


# --------------------------------------------------------------------------- skeleton

_REWRITES = (("tech", "tek"), ("tion", "shan"), ("sion", "shan"), ("tur", "chur"), ("x", "ks"), ("qu", "k"),
             ("ck", "k"))
_DIGRAPHS = (("chh", "j"), ("ch", "j"), ("sh", "s"), ("ph", "p"), ("th", "t"), ("dh", "t"),
             ("kh", "k"), ("gh", "k"), ("bh", "p"), ("jh", "j"))
_SOFT_C = re.compile(r"c(?=[eiy])")
_SOFT_G = re.compile(r"(?<=.)g(?=[eiy])")  # not word-initial: "global", but "technologies", "engineering"
_CLASSES = str.maketrans({"c": "k", "g": "k", "q": "k", "d": "t", "b": "p", "f": "p", "w": "v", "z": "s"})
_NON_LATIN = re.compile(r"[^a-z]")
_REPEATS = re.compile(r"(.)\1+")
VOWELS = frozenset("aeiou")


@lru_cache(maxsize=1 << 20)
def skeleton_token(token):
    """Consonant-class key of one Latin (or romanized) word; "" if too short."""
    t = _NON_LATIN.sub("", token.lower())
    if not t:
        return ""
    for a, b in _REWRITES:
        t = t.replace(a, b)
    for a, b in _DIGRAPHS:
        t = t.replace(a, b)
    t = _SOFT_G.sub("j", _SOFT_C.sub("s", t)).translate(_CLASSES)
    first = t[0]
    head = "a" if first in VOWELS else first
    body = "".join(ch for ch in t[1:] if ch not in VOWELS and ch not in "yh")
    key = _REPEATS.sub(r"\1", head + body)
    while len(key) >= 4 and key[-1] in "sj":  # plural: -s / -ज / -ஸ
        key = key[:-1]
    return key if len(key) >= 2 else ""


def name_skeleton(name):
    """Space-joined skeleton keys of the words of a name (Indic words are romanized first)."""
    return " ".join(k for k in (skeleton_token(w) for w in romanize(name).split()) if k)


def skeletons(values):
    """``name_skeleton`` over a list of strings, computing each distinct value once."""
    cache = {}
    out = []
    for v in values:
        s = cache.get(v)
        if s is None:
            s = cache[v] = name_skeleton(v or "")
        out.append(s)
    return out


# --------------------------------------------------------------------------- exact skeleton key channel

INDIC_KEY_FIELD = "name_skel_key"


def skeleton_key(skeleton):
    """Sorted unique skeleton keys of one name as a single exact-match key ("" if none)."""
    return " ".join(sorted(set(skeleton.split())))


def skeleton_keys(names):
    """``skeleton_key(name_skeleton(name))`` for a list of names (each distinct value once)."""
    return [skeleton_key(s) for s in skeletons(names)]


def build_indic_key_pools(names, countries, scripts, country_order):
    """Per country, a ``CountryPool`` whose ``name_skel_key`` field indexes Indic/mixed targets.

    ``names`` (list of ``name_stripped``), ``countries`` and ``scripts`` (``name_script``
    labels) cover the whole target pool in its global order. Only targets whose name script
    is Indic or mixed are indexed: Phase 3 measured that Latin S1 names match those through
    the whole-name skeleton key, while single skeleton tokens are too common to be useful.
    """
    import numpy as np
    import pyarrow as pa

    from ..normalize.text import INDIC_SCRIPTS
    from ..tokens import CountryPool, build_field_index

    indic = {label for _, _, label in INDIC_SCRIPTS} | {"mixed"}
    countries = np.asarray(countries, dtype=object)
    eligible = np.isin(np.asarray(scripts, dtype=object), list(indic))
    pools = {}
    for c in country_order:
        idx = np.flatnonzero(eligible & (countries == c)).astype(np.int32)
        keys = skeleton_keys([names[i] for i in idx])
        cp = CountryPool(country=c, target_idx=idx)
        cp.fields[INDIC_KEY_FIELD] = build_field_index(pa.array(keys, pa.string()), "key")
        pools[c] = cp
    return pools
