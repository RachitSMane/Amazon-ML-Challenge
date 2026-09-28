"""Explicit, auditable legal-form lists and their removal from normalized names.

Every entry is a token sequence as it looks **after** ``text.clean`` + ``text.fold`` (lower
case, no accents, punctuation turned into spaces, dotted initialisms joined), mapped to a
canonical label. Nothing outside these lists is ever removed.

Rules applied by ``strip_legal_forms``:

* Suffix forms are removed only at the **end** of the name, prefix forms only at the
  **start**; a listed word in the middle of a name is kept (``limited edition cakes``).
* Removal repeats, longest sequence first, so ``pvt ltd``, ``private limited co`` or
  ``opc pvt ltd`` combinations are all handled.
* After a suffix is removed, a dangling connector (``and``/``et``) is removed as well
  (``smith and co`` -> ``smith``).
* A name is never reduced to nothing: a form is kept if removing it would empty the name.

Which lists apply depends on the country, which stays an open-set string:

* ``GLOBAL`` applies to every country, including countries never seen in training. It
  holds the international English forms (ltd, inc, corp, llc, llp, plc, co) and the Indic-
  script spellings of private/limited/LLP, which are unambiguous in any country.
* ``BY_COUNTRY[country]`` adds country-specific forms for US, India and France.

The Indic spellings were taken from the most frequent trailing tokens of Indic-script
names in train S2/S3 (e.g. Devanagari ``प्राइवेट लिमिटेड`` 246k, ``प्रा लि`` 55k, ``एलएलपी`` 24k),
written without zero-width joiners because ``text.clean`` removes them.
"""

from functools import lru_cache

# --------------------------------------------------------------------------- the lists

# canonical label -> spellings (space-separated token sequences)
_GLOBAL_SUFFIX = {
    "ltd": ["ltd", "limited"],
    "inc": ["inc", "incorporated"],
    "corp": ["corp", "corporation"],
    "co": ["co", "company"],
    "llc": ["llc", "l l c"],
    "llp": ["llp", "l l p"],
    "plc": ["plc"],
    "pvt": ["pvt", "private"],
    "co ltd": ["co ltd", "company limited"],
    "pvt ltd": ["pvt ltd", "private ltd", "pvt limited", "private limited", "p ltd", "p limited", "pvt co"],
    # Indic scripts: private, limited, pvt/ltd abbreviations, LLP.
    "pvt ltd@indic": [
        "प्राइवेट लिमिटेड", "प्रा लि",                  # Devanagari
        "প্রাইভেট লিমিটেড",                             # Bengali
        "ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ", "ਪ੍ਰਾ ਲਿ",                  # Gurmukhi
        "પ્રાઇવેટ લિમિટેડ", "પ્રા લિ",                  # Gujarati
        "ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍",                           # Oriya
        "பிரைவேட் லிமிடெட்",                           # Tamil
        "ప్రైవేట్ లిమిటెడ్",                            # Telugu
        "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",                           # Kannada
        "പ്രൈവറ്റ് ലിമിറ്റഡ്",                          # Malayalam
    ],
    "ltd@indic": [
        "लिमिटेड", "लि", "লিমিটেড", "ਲਿਮਟਿਡ", "ਲਿ", "લિમિટેડ", "લિ",
        "ଲିମିଟେଡ୍", "லிமிடெட்", "లిమిటెడ్", "ಲಿಮಿಟೆಡ್", "ലിമിറ്റഡ്",
    ],
    "pvt@indic": [
        "प्राइवेट", "প্রাইভেট", "ਪ੍ਰਾਈਵੇਟ", "પ્રાઇવેટ", "ପ୍ରାଇଭେଟ୍",
        "பிரைவேட்", "ప్రైవేట్", "ಪ್ರೈವೇಟ್", "പ്രൈവറ്റ്",
    ],
    "llp@indic": [
        "एलएलपी", "এলএলপি", "ਐਲਐਲਪੀ", "એલએલપી", "ଏଲ୍ଏଲ୍ପି",
        "எல்எல்பி", "ఎల్ఎల్పీ", "ಎಲ್ಎಲ್ಪಿ", "എൽഎൽപി",
    ],
}

_BY_COUNTRY_SUFFIX = {
    "US": {
        "lp": ["lp", "l p"],
        "pllc": ["pllc", "p l l c"],
        "lllp": ["lllp"],
    },
    "India": {
        "opc": ["opc", "one person company"],
    },
    "France": {
        # Sociétés commerciales, civiles et d'exercice libéral; entreprise individuelle.
        "sarl": ["sarl"], "sas": ["sas"], "sasu": ["sasu"], "sa": ["sa"], "eurl": ["eurl"],
        "sci": ["sci"], "snc": ["snc"], "scs": ["scs"], "sca": ["sca"], "scop": ["scop"],
        "scp": ["scp"], "scm": ["scm"], "selarl": ["selarl"], "selas": ["selas"],
        "selafa": ["selafa"], "sem": ["sem"], "gie": ["gie"], "eirl": ["eirl"], "ei": ["ei"],
        "cie": ["cie", "compagnie", "et cie"],
    },
}

# Prefix forms (only removed at the start of a name).
_BY_COUNTRY_PREFIX = {
    "India": {
        "m/s": ["m s", "messrs"],  # "M/s ABC Traders" (Messrs), after "/" became a space
    },
    "France": {
        "sarl": ["sarl"], "sas": ["sas"], "sasu": ["sasu"], "sa": ["sa"], "eurl": ["eurl"],
        "sci": ["sci"], "snc": ["snc"], "selarl": ["selarl"], "selas": ["selas"], "gie": ["gie"],
        # Designators rather than legal forms, but dropped in the same way.
        "ets": ["ets", "etablissements", "etablissement"],
        "ste": ["ste", "societe"],
    },
}

CONNECTORS = frozenset({"and", "et"})


def _to_rules(mapping):
    """{label: [spelling, ...]} -> [(tokens, label), ...]; an ``@script`` label tag is dropped."""
    rules = []
    for label, spellings in mapping.items():
        for spelling in spellings:
            rules.append((tuple(spelling.split()), label.split("@")[0]))
    return rules


def legal_form_table(country):
    """All (tokens, label) suffix and prefix rules that apply to ``country`` (for audits)."""
    suffix = _to_rules(_GLOBAL_SUFFIX) + _to_rules(_BY_COUNTRY_SUFFIX.get(country, {}))
    prefix = _to_rules(_BY_COUNTRY_PREFIX.get(country, {}))
    return suffix, prefix


@lru_cache(maxsize=None)
def _index(country):
    """Rules indexed by the token they must end (suffix) or start (prefix) with."""
    suffix, prefix = legal_form_table(country)
    by_last, by_first = {}, {}
    for tokens, label in suffix:
        by_last.setdefault(tokens[-1], []).append((tokens, label))
    for tokens, label in prefix:
        by_first.setdefault(tokens[0], []).append((tokens, label))
    for table in (by_last, by_first):
        for rules in table.values():
            rules.sort(key=lambda rule: -len(rule[0]))
    return by_last, by_first


def strip_legal_forms(name, country=""):
    """Return ``(stripped_name, forms)`` for a cleaned, folded name.

    ``forms`` lists the canonical labels removed, prefixes first, then suffixes from the
    innermost outwards (the order they appear in the name).
    """
    tokens = name.split()
    if len(tokens) < 2:
        return name, []
    by_last, by_first = _index(country)
    prefixes, suffixes = [], []

    changed = True
    while changed:
        changed = False
        for tokens_rule, label in by_last.get(tokens[-1], ()):
            n = len(tokens_rule)
            if len(tokens) > n and tuple(tokens[-n:]) == tokens_rule:
                tokens = tokens[:-n]
                suffixes.append(label)
                if len(tokens) > 1 and tokens[-1] in CONNECTORS:
                    tokens = tokens[:-1]
                changed = True
                break

    changed = True
    while changed:
        changed = False
        for tokens_rule, label in by_first.get(tokens[0], ()):
            n = len(tokens_rule)
            if len(tokens) > n and tuple(tokens[:n]) == tokens_rule:
                tokens = tokens[n:]
                prefixes.append(label)
                changed = True
                break

    return " ".join(tokens), prefixes + suffixes[::-1]
