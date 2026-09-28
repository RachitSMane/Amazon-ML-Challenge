"""Unit tests for src/ber/normalize (text, legal forms, addresses, scripts, normalized cache).

Run from code/business_entity_resolution/:

    python -m unittest tests.test_normalize -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ber import io  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.normalize import NORMALIZED_SCHEMA, normalize_name, normalize_table  # noqa: E402
from ber.normalize import cache as norm_cache  # noqa: E402
from ber.normalize.address import normalize_address  # noqa: E402
from ber.normalize.legal_forms import strip_legal_forms  # noqa: E402
from ber.normalize.text import clean, fold, is_placeholder, script_of  # noqa: E402
from test_io import FILES, HEADER, file_hashes, write_dataset  # noqa: E402


class TextTest(unittest.TestCase):
    def test_nfkc_casefold_whitespace(self):
        self.assertEqual(clean("ＡＢＣ　Ｔｒａｄｅｒｓ"), "abc traders")
        self.assertEqual(clean("  Acme\t\tHOLDINGS \n"), "acme holdings")
        self.assertEqual(clean("Straße"), "strasse")

    def test_punctuation_apostrophes_ampersand(self):
        self.assertEqual(clean("McDonald's"), "mcdonalds")
        self.assertEqual(clean("McDonald’s Corp."), "mcdonalds corp")
        self.assertEqual(clean("Smith & Sons"), "smith and sons")
        self.assertEqual(clean("Appel & Fils", "France"), "appel et fils")
        self.assertEqual(clean("ABC-Tech/Labs (India) [OPC]"), "abc tech labs india opc")
        self.assertEqual(clean("foo_bar"), "foo bar")

    def test_french_elision(self):
        self.assertEqual(clean("L'Atelier d'Anna", "France"), "l atelier d anna")
        self.assertEqual(clean("L'Atelier", "US"), "latelier")

    def test_initialisms(self):
        self.assertEqual(clean("P.V.T. Ltd."), "pvt ltd")
        self.assertEqual(clean("U.S.A Foods"), "usa foods")
        self.assertEqual(clean("Dr. Smith"), "dr smith")

    def test_accents_folded_only_in_latin(self):
        self.assertEqual(clean("Établissements Dëleves"), "établissements dëleves")
        self.assertEqual(fold(clean("Établissements Dëleves")), "etablissements deleves")
        self.assertEqual(fold("cœur ærø"), "coeur aero")
        hindi = "राम मार्केटिंग"
        self.assertEqual(fold(clean(hindi)), hindi)

    def test_indic_text_preserved(self):
        # Vowel signs and viramas are combining marks; they must survive punctuation removal.
        self.assertEqual(clean("ग्लोबल इन्वेस्टमेंट प्रा. लि."), "ग्लोबल इन्वेस्टमेंट प्रा लि")
        self.assertEqual(clean("குளோபல் பிசினஸ்"), "குளோபல் பிசினஸ்")

    def test_zero_width_removed(self):
        self.assertEqual(clean("ଏଲ୍‌ଏଲ୍‌ପି"), "ଏଲ୍ଏଲ୍ପି")
        self.assertEqual(clean("ab​c­"), "abc")

    def test_placeholders(self):
        for value in ("", "   ", "<NULL>", "null", "NULL", " None ", "n/a"):
            self.assertTrue(is_placeholder(value), value)
            self.assertEqual(clean(value), "")
        self.assertFalse(is_placeholder("Nullabor Traders"))
        self.assertEqual(clean(None), "")

    def test_deterministic(self):
        self.assertEqual(clean("Café & Co., L'Été", "France"), clean("Café & Co., L'Été", "France"))


class ScriptTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(script_of("acme 123"), "latin")
        self.assertEqual(script_of("établissements"), "latin")
        self.assertEqual(script_of("123 456"), "empty")
        self.assertEqual(script_of(""), "empty")
        self.assertEqual(script_of("राम मार्केटिंग"), "devanagari")
        self.assertEqual(script_of("প্রাইভেট"), "bengali")
        self.assertEqual(script_of("பிரைவேட்"), "tamil")
        self.assertEqual(script_of("ప్రైవేట్"), "telugu")
        self.assertEqual(script_of("royal हॉस्पिटैलिटी"), "mixed")
        self.assertEqual(script_of("東京"), "other")
        self.assertEqual(script_of("مرحبا"), "other")


class LegalFormTest(unittest.TestCase):
    def check(self, raw, country, stripped, forms):
        name = normalize_name(raw, country)
        self.assertEqual((name.stripped, name.legal_forms), (stripped, forms), raw)

    def test_us(self):
        self.check("ACME Corp., Inc.", "US", "acme", "corp inc")
        self.check("Acme Holdings LLC", "US", "acme holdings", "llc")
        self.check("Acme L.L.C.", "US", "acme", "llc")
        self.check("Smith & Co.", "US", "smith", "co")
        self.check("Blue River LP", "US", "blue river", "lp")
        self.check("The Coca-Cola Company", "US", "the coca cola", "co")

    def test_india(self):
        self.check("Sharma Traders Pvt. Ltd.", "India", "sharma traders", "pvt ltd")
        self.check("Sharma Traders Private Limited", "India", "sharma traders", "pvt ltd")
        self.check("Sharma Traders (P) Ltd", "India", "sharma traders", "pvt ltd")
        self.check("M/s Sharma Traders", "India", "sharma traders", "m/s")
        self.check("Tech Labs (OPC) Private Limited", "India", "tech labs", "opc pvt ltd")
        self.check("Green Farms LLP", "India", "green farms", "llp")

    def test_indic_scripts(self):
        self.check("राम मार्केटिंग प्राइवेट लिमिटेड", "India", "राम मार्केटिंग", "pvt ltd")
        self.check("ग्लोबल इन्वेस्टमेंट प्रा. लि.", "India", "ग्लोबल इन्वेस्टमेंट", "pvt ltd")
        self.check("आदित्य प्रॉपर्टीज एलएलपी", "India", "आदित्य प्रॉपर्टीज", "llp")
        self.check("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "India", "குளோபல் பிசினஸ்", "pvt ltd")
        self.check("Royal हॉस्पिटैलिटी Private Limited", "India", "royal हॉस्पिटैलिटी", "pvt ltd")

    def test_france(self):
        self.check("Bordeaux Club SARL", "France", "bordeaux club", "sarl")
        self.check("SCI Ptit Àmicale", "France", "ptit amicale", "sci")
        self.check("QHC Culture [EURL]", "France", "qhc culture", "eurl")
        self.check("Établissements Dëleves EURL", "France", "deleves", "ets eurl")
        self.check("Société Martin SAS", "France", "martin", "ste sas")
        self.check("Martin et Cie", "France", "martin", "cie")

    def test_country_specific_lists(self):
        # French forms apply only to France; global forms apply to any country, even unseen ones.
        self.check("Bordeaux Club SARL", "US", "bordeaux club sarl", "")
        self.check("Nordic Trade Ltd", "Norway", "nordic trade", "ltd")
        self.check("Nordic Trade SARL", "Norway", "nordic trade sarl", "")

    def test_only_listed_words_at_the_edges(self):
        self.check("Limited Edition Cakes", "US", "limited edition cakes", "")
        self.check("Private Eye Detectives", "India", "private eye detectives", "")
        self.check("Company Store Outlet", "US", "company store outlet", "")

    def test_never_empty(self):
        self.check("Limited", "US", "limited", "")
        self.check("SARL", "France", "sarl", "")
        self.check("Pvt Ltd", "India", "pvt", "ltd")

    def test_strip_is_idempotent(self):
        stripped, _ = strip_legal_forms("sharma traders pvt ltd", "India")
        self.assertEqual(strip_legal_forms(stripped, "India"), (stripped, []))

    def test_name_fields(self):
        name = normalize_name("Café Crème SARL", "France")
        self.assertEqual(name.norm, "café crème sarl")
        self.assertEqual(name.fold, "cafe creme sarl")
        self.assertEqual(name.key, "cafe creme")
        self.assertEqual(name.script, "latin")


class AddressTest(unittest.TestCase):
    def test_missing(self):
        for raw in ("", "   ", "<NULL>", "NULL", " , NULL ,", "<NULL>, <NULL>", "null , ,"):
            a = normalize_address(raw, "US")
            self.assertTrue(a.missing, raw)
            self.assertEqual((a.norm, a.postcode, a.house_numbers, a.script), ("", "", "", "empty"))
        self.assertEqual(normalize_address(None, "US").missing, True)

    def test_placeholder_components_removed(self):
        a = normalize_address("2612 SOUTHVIEW COURT, <NULL>, BRAINERD, MN", "US")
        self.assertFalse(a.missing)
        self.assertEqual(a.norm, "2612 southview court brainerd mn")
        self.assertEqual(a.ws, "2612 SOUTHVIEW COURT, <NULL>, BRAINERD, MN")
        self.assertEqual(normalize_address("22120 COYOTE CAVE TRL, null, SPICEWOOD, TX", "US").norm,
                         "22120 coyote cave trail spicewood tx")

    def test_whitespace_and_punctuation_versions(self):
        a = normalize_address("  12,  MG   Road ;  Bangalore  ", "India")
        self.assertEqual(a.ws, "12, MG Road ; Bangalore")
        self.assertEqual(a.punct, "12 mg road bangalore")

    def test_us_abbreviations_and_numbers(self):
        a = normalize_address("123 Main St Ste 200, Springfield, IL 62704-1234", "US")
        self.assertEqual(a.norm, "123 main street suite 200 springfield il 62704 1234")
        self.assertEqual((a.postcode, a.house_numbers, a.unit_numbers), ("62704", "123", "200"))
        self.assertEqual(normalize_address("415 Northfield Blvd, Unit A11, Murfreesboro, TN", "US").unit_numbers, "a11")
        self.assertEqual(normalize_address("9 Oak Ave Apt #5, Reno, NV", "US").unit_numbers, "5")

    def test_nebraska_not_expanded(self):
        self.assertEqual(normalize_address("<NULL>, NE, OMAHA, 10333- BROADMOOR CT", "US").norm,
                         "ne omaha 10333 broadmoor court")

    def test_us_house_number_is_not_a_postcode(self):
        a = normalize_address("17200 Bell Rd, Surprise, AZ", "US")
        self.assertEqual((a.postcode, a.house_numbers), ("", "17200"))

    def test_zero_padding_and_hashes(self):
        self.assertEqual(normalize_address("06252 Taliaferro Way, Alexandria, VA", "US").house_numbers, "6252")
        self.assertEqual(normalize_address("##8 Willow Oak Lane, Saint Louis, MO", "US").house_numbers, "8")
        self.assertEqual(normalize_address("024 av gustave delory, roubaix", "France").norm, "24 avenue gustave delory roubaix")

    def test_india(self):
        a = normalize_address("Plot No.: 3829/3250, Near Metro Pillar No: 234, Kanhaiya Nagar, Delhi - 110035", "India")
        self.assertEqual((a.postcode, a.house_numbers), ("110035", "3829-3250 234"))
        self.assertEqual(normalize_address("H. No. 154 - A, Veer Savarkar Nagar, Bareilly", "India").house_numbers, "154")
        self.assertEqual(normalize_address("G-58/1, Benson Cross Road, Bangalore", "India").house_numbers, "g-58-1")
        a = normalize_address("Shop No. 188, 1St Floor, Powai Plaza, Mumbai 400 076", "India")
        self.assertEqual((a.unit_numbers, a.house_numbers, a.postcode), ("188", "", "400076"))
        self.assertEqual(normalize_address("Nr Bus Stn, Opp SBI, Sec 5", "India").norm,
                         "near bus station opposite sbi sector 5")

    def test_india_indic_address_kept(self):
        a = normalize_address("PLOT NO B-78/1, MIDC ANAND NAGAR, THANE, महाराष्ट्र", "India")
        self.assertEqual(a.norm, "plot no b 78 1 midc anand nagar thane महाराष्ट्र")
        self.assertEqual((a.script, a.house_numbers), ("mixed", "b-78-1"))

    def test_france(self):
        a = normalize_address("17 BIS RUE MAURICE TERRIEN, NANTES", "France")
        self.assertEqual((a.norm, a.house_numbers), ("17 bis rue maurice terrien nantes", "17bis"))
        a = normalize_address("N°24 R DE LA TRANQUILITE, 59200 TOURCOING", "France")
        self.assertEqual((a.norm, a.postcode, a.house_numbers), ("no 24 rue de la tranquilite 59200 tourcoing", "59200", "24"))
        a = normalize_address("Nº 20 R. D'ALZON, BORDEAUX", "France")
        self.assertEqual((a.norm, a.house_numbers), ("no 20 rue d alzon bordeaux", "20"))
        self.assertEqual(normalize_address("12 BD VAUBAN, STE CATHERINE, LILLE", "France").norm,
                         "12 boulevard vauban sainte catherine lille")
        self.assertEqual(normalize_address("5 ALLÉE DES HÊTRES, PORNIC", "France").norm, "5 allee des hetres pornic")

    def test_unknown_country_uses_generic_rules(self):
        a = normalize_address("12 Harbour Rd, Oslo, 0150", "Norway")
        self.assertEqual((a.norm, a.postcode, a.house_numbers), ("12 harbour road oslo 150", "0150", "12"))


def raw_table(rows):
    """Arrow table with the raw-cache columns from (id, name, address, country, source) rows."""
    cols = list(zip(*rows))
    return pa.table({
        "entity_id": pa.array(cols[0], pa.string()), "business_name": pa.array(cols[1], pa.string()),
        "business_address": pa.array(cols[2], pa.string()), "country": pa.array(cols[3], pa.string()),
        "source": pa.array(cols[4], pa.int8()),
    })


class NormalizeTableTest(unittest.TestCase):
    def test_schema_and_passthrough(self):
        table = raw_table([
            ("S2-1", "Acme Inc", "1 Main St, <NULL>, Reno, NV", "US", 2),
            ("S2-2", "राम मार्केटिंग प्राइवेट लिमिटेड", "", "India", 2),
            ("S2-3", "Bordeaux Club SARL", "3 R. de la Paix, Lille", "France", 2),
        ])
        out = normalize_table(table)
        self.assertEqual(out.schema, NORMALIZED_SCHEMA)
        self.assertEqual(out["entity_id"].to_pylist(), ["S2-1", "S2-2", "S2-3"])
        self.assertEqual(out["country"].to_pylist(), ["US", "India", "France"])
        self.assertEqual(out["name_raw"].to_pylist(), table["business_name"].to_pylist())
        self.assertEqual(out["address_raw"].to_pylist(), table["business_address"].to_pylist())
        self.assertEqual(out["address_missing"].to_pylist(), [False, True, False])
        self.assertEqual(out["name_script"].to_pylist(), ["latin", "devanagari", "latin"])
        self.assertEqual(out["legal_forms"].to_pylist(), ["inc", "pvt ltd", "sarl"])
        self.assertEqual(out["address_norm"].to_pylist()[2], "3 rue de la paix lille")
        self.assertEqual(out.column("address_norm").null_count, 0)


NORM_FILES = dict(FILES)
NORM_FILES["train/train_source2.tsv"] = (
    HEADER
    + "S2-10\tACME INC.\t1 main street\tUS\n"
    + "S2-11\tQuoted Traders \"Pvt\"\t<NULL>\tIndia\n"
    + "S2-12\tराम मार्केटिंग प्राइवेट लिमिटेड\tPLOT NO B-78/1, THANE, महाराष्ट्र\tIndia\n"
)


class NormCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = write_dataset(Path(self._tmp.name) / "dataset", NORM_FILES)
        self.cfg = Config.discover(self.root, Path(self._tmp.name) / "work")
        io.build_raw_cache(self.cfg)

    def tearDown(self):
        self._tmp.cleanup()

    def test_requires_raw_cache(self):
        (self.cfg.raw_cache_dir / "manifest.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "raw cache"):
            norm_cache.build_norm_cache(self.cfg, workers=1)

    def test_round_trip(self):
        manifest = norm_cache.build_norm_cache(self.cfg, workers=1, batch_rows=2)
        self.assertEqual(sorted(manifest["entries"]), sorted(norm_cache.source_keys()))
        for key in norm_cache.source_keys():
            split, source = key.split("_source")
            raw = io.read_source_table(self.cfg, split, int(source))
            norm = norm_cache.read_normalized_table(self.cfg, split, int(source))
            self.assertEqual(norm.schema, NORMALIZED_SCHEMA)
            self.assertEqual(norm["entity_id"].to_pylist(), raw["entity_id"].to_pylist())
            self.assertEqual(norm["country"].to_pylist(), raw["country"].to_pylist())
            self.assertEqual(norm["source"].to_pylist(), raw["source"].to_pylist())
            self.assertEqual(norm["name_raw"].to_pylist(), raw["business_name"].to_pylist())
            self.assertEqual(norm["address_raw"].to_pylist(), raw["business_address"].to_pylist())
            self.assertTrue(norm.equals(normalize_table(raw)), key)
        df = norm_cache.load_normalized(self.cfg, "train", 2)
        self.assertEqual(df["address_missing"].tolist(), [False, True, False])
        self.assertEqual(df["name_script"].tolist(), ["latin", "latin", "devanagari"])
        self.assertEqual(df["house_numbers"].tolist(), ["1", "", "b-78-1"])
        self.assertEqual(sum(len(c) for c in norm_cache.iter_normalized_chunks(self.cfg, "train", 2, batch_rows=2)), 3)

    def test_multiprocess_equals_single_process(self):
        norm_cache.build_norm_cache(self.cfg, workers=1)
        single = norm_cache.read_normalized_table(self.cfg, "train", 2)
        norm_cache.build_norm_cache(self.cfg, workers=2, batch_rows=1, force=True)
        self.assertTrue(norm_cache.read_normalized_table(self.cfg, "train", 2).equals(single))

    def test_raw_files_unchanged(self):
        before = file_hashes(self.root)
        raw_before = {p: p.read_bytes() for p in self.cfg.raw_cache_dir.glob("*.parquet")}
        norm_cache.build_norm_cache(self.cfg, workers=1)
        self.assertEqual(file_hashes(self.root), before)
        self.assertEqual({p: p.read_bytes() for p in self.cfg.raw_cache_dir.glob("*.parquet")}, raw_before)
        self.assertEqual(list(self.cfg.norm_cache_dir.glob("*.tmp")), [])

    def test_fresh_entries_skipped(self):
        norm_cache.build_norm_cache(self.cfg, workers=1)
        messages = []
        norm_cache.build_norm_cache(self.cfg, workers=1, log=messages.append)
        self.assertEqual(len(messages), 6)
        self.assertTrue(all("skipped" in m for m in messages))

    def test_invalidated_by_raw_change(self):
        norm_cache.build_norm_cache(self.cfg, workers=1)
        (self.root / "train/train_source3.tsv").write_text(
            HEADER + "S3-20\tAcme\t1 Main St\tUS\nS3-21\tNew Co\t2 Elm St\tUS\n", encoding="utf-8", newline="\n")
        self.assertFalse(norm_cache.is_norm_fresh(self.cfg, "train_source3"))  # raw TSV changed
        io.build_raw_cache(self.cfg)
        self.assertFalse(norm_cache.is_norm_fresh(self.cfg, "train_source3"))  # raw cache rebuilt
        self.assertTrue(norm_cache.is_norm_fresh(self.cfg, "train_source1"))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            norm_cache.read_normalized_table(self.cfg, "train", 3)
        norm_cache.build_norm_cache(self.cfg, workers=1)
        self.assertEqual(len(norm_cache.read_normalized_table(self.cfg, "train", 3)), 2)

    def test_invalidated_by_code_change(self):
        norm_cache.build_norm_cache(self.cfg, workers=1)
        with mock.patch.object(norm_cache, "code_fingerprint", return_value="different"):
            self.assertFalse(norm_cache.is_norm_fresh(self.cfg, "train_source1"))

    def test_invalidated_by_version_change(self):
        norm_cache.build_norm_cache(self.cfg, workers=1)
        with mock.patch.object(norm_cache, "NORM_CACHE_VERSION", 999):
            self.assertFalse(norm_cache.is_norm_fresh(self.cfg, "train_source1"))


if __name__ == "__main__":
    unittest.main()
