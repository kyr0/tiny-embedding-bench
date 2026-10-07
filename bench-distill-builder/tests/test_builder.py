import copy
import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import build_dataset as b


def fixture():
    languages = ["eng_Latn", "jpn_Jpan"]
    raw = {}
    for lang in languages:
        rows = []
        for i in range(6):
            for number in (1, 2):
                rows.append({"link": f"https://example.test/topic-{i}", "question_number": number,
                             "dialect": lang,
                             "flores_passage": f"  {lang}: {i} 雪\n" + "abc" * i,
                             "question": f"{lang} Topic {i}, question {number}? " + "q" * number})
        if lang == "jpn_Jpan":
            rows.reverse()
        raw[lang] = b"".join(b.packed(row) + b"\n" for row in rows)
    lock = {"repo_id": "facebook/belebele", "revision": "a" * 40, "split": "test",
            "source_rows_per_language": 12, "source_passages_per_language": 6,
            "languages": [{"code": l, "name": l} for l in languages],
            "files": {f"data/{l}.jsonl": {"sha256": b.sha(v), "bytes": len(v)} for l, v in raw.items()}}
    en = [json.loads(line) for line in raw["eng_Latn"].splitlines()]
    allow = {"source_revision": lock["revision"], "approved": [
        {"source_url": r["link"], "question_number": r["question_number"], "english": r["question"]}
        for r in en]}
    return raw, lock, allow


def mutate_source(raw, lock, language, change):
    raw, lock = copy.deepcopy(raw), copy.deepcopy(lock)
    rows = [json.loads(line) for line in raw[language].splitlines()]
    change(rows)
    value = b"".join(b.packed(row) + b"\n" for row in rows)
    raw[language] = value
    lock["files"][f"data/{language}.jsonl"] = {"bytes": len(value), "sha256": b.sha(value)}
    return raw, lock


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.lock, self.allow = fixture()
        self.indexed = b.index_sources(self.raw, self.lock)
        self.data = b.make_dataset(self.indexed, self.lock, self.allow, 4, 2)

    def test_alignment_uses_source_keys_and_preserves_original_text(self):
        self.assertEqual(len(self.data["corpus"]), 4)
        relations = {q: p for q, p, _ in self.data["qrels"]}
        corpus = {r["id"]: r for r in self.data["corpus"]}
        for row in self.data["queries"]:
            url = corpus[relations[row["id"]]]["source_url"]
            for i, language in enumerate(self.data["languages"]):
                source = self.indexed[language]["questions"][url, row["question_number"]]
                self.assertEqual(row["text"][i], source["question"])
                self.assertEqual(corpus[relations[row["id"]]]["text"][i], source["flores_passage"])

    def test_deterministic_under_source_and_allowlist_reordering(self):
        raw, lock = mutate_source(self.raw, self.lock, "eng_Latn", lambda rows: rows.reverse())
        allow = copy.deepcopy(self.allow)
        allow["approved"].reverse()
        urls1, keys1 = b.select(self.indexed, self.lock, self.allow, 4, 2)
        urls2, keys2 = b.select(b.index_sources(raw, lock), lock, allow, 4, 2)
        self.assertEqual((urls1, keys1), (urls2, keys2))

    def test_bad_cached_source_hash(self):
        raw = dict(self.raw)
        raw["eng_Latn"] = raw["eng_Latn"].replace(b"topic-0", b"topic-X")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            b.index_sources(raw, self.lock)

    def test_duplicate_key_is_rejected(self):
        raw, lock = mutate_source(self.raw, self.lock, "eng_Latn", lambda rows: rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate source question"):
            b.index_sources(raw, lock)

    def test_inconsistent_passage_for_same_url_is_rejected(self):
        raw, lock = mutate_source(self.raw, self.lock, "eng_Latn",
                                  lambda rows: rows[1].update(flores_passage="different"))
        with self.assertRaisesRegex(ValueError, "Inconsistent passage"):
            b.index_sources(raw, lock)

    def test_missing_translation_key_is_rejected(self):
        raw, lock = mutate_source(self.raw, self.lock, "jpn_Jpan",
                                  lambda rows: rows[0].update(question_number=3))
        with self.assertRaisesRegex(ValueError, "Cross-language"):
            b.index_sources(raw, lock)

    def test_wrong_dialect_is_rejected(self):
        raw, lock = mutate_source(self.raw, self.lock, "jpn_Jpan",
                                  lambda rows: rows[0].update(dialect="eng_Latn"))
        with self.assertRaisesRegex(ValueError, "Wrong dialect"):
            b.index_sources(raw, lock)

    def test_curation_text_is_checked_and_quota_never_silently_reduced(self):
        changed = copy.deepcopy(self.allow)
        changed["approved"][0]["english"] = "Rewritten question"
        with self.assertRaisesRegex(ValueError, "Curation text changed"):
            b.select(self.indexed, self.lock, changed, 4, 2)
        with self.assertRaisesRegex(ValueError, "screened passage groups"):
            b.select(self.indexed, self.lock, self.allow, 7, 7)
        with self.assertRaisesRegex(ValueError, "Insufficient passage"):
            b.select(self.indexed, self.lock, self.allow, 7, 2)

    def test_utf8_bytes_not_character_count(self):
        indexed = copy.deepcopy(self.indexed)
        urls = list(indexed["eng_Latn"]["passages"])
        for language in indexed:
            for i, url in enumerate(urls):
                indexed[language]["passages"][url] = "x" * (100 + i)
            indexed[language]["passages"][urls[0]] = "雪" * 10  # 10 chars, 30 bytes.
            indexed[language]["passages"][urls[1]] = "a" * 20  # 20 chars, 20 bytes.
        allow = copy.deepcopy(self.allow)
        allow["approved"] = [e for e in allow["approved"] if e["source_url"] == urls[5]]
        selected, _ = b.select(indexed, self.lock, allow, 2, 1)
        self.assertIn(urls[1], selected)
        self.assertNotIn(urls[0], selected)

    def test_invalid_qrels_and_counts_are_rejected(self):
        for change in (
            lambda d: d["qrels"][0].__setitem__(1, "missing"),
            lambda d: d["qrels"][0].__setitem__(2, True),
            lambda d: d["qrels"].pop(),
            lambda d: d["corpus"].pop(),
            lambda d: d["corpus"][0]["text"].pop(),
        ):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(ValueError):
                    b.validate_dataset(data)

    def test_mono_cross_and_mixed_qrels(self):
        for target, candidates, positives in ((None, 4, 1), ("jpn_Jpan", 4, 1), ("*", 8, 2)):
            corpus, queries, qrels = b.task(self.data, "eng_Latn", target)
            self.assertEqual(len(corpus), candidates)
            self.assertEqual(len(queries), 2)
            self.assertEqual(set(queries), set(qrels))
            for rels in qrels.values():
                self.assertEqual(len(rels), positives)
                self.assertTrue(set(rels) <= set(corpus))
                self.assertEqual(set(rels.values()), {1})

    def test_output_byte_limit_is_strict_and_includes_newline(self):
        size = len(b.packed(self.data)) + 1
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "out"
            with self.assertRaisesRegex(ValueError, "requires <"):
                b.write_dataset(self.data, self.lock, output, max_bytes=size)
            self.assertFalse(output.exists())
            b.write_dataset(self.data, self.lock, output, max_bytes=size + 1)
            self.assertEqual((output / "dataset.json").stat().st_size, size)

    def test_release_fingerprint_is_independent_of_output_manifest(self):
        raw = b.packed(self.data) + b"\n"
        expected = {"dataset.json": {"bytes": len(raw), "sha256": b.sha(raw)}}
        with patch.object(b, "read_json", return_value=expected):
            b.check_release(self.data)
            changed = copy.deepcopy(self.data)
            changed["corpus"][0]["text"][0] += "Changed"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                b.check_release(changed)

    def test_rebuilds_are_identical_and_verify_detects_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            a, c = Path(root) / "a", Path(root) / "c"
            b.write_dataset(self.data, self.lock, a)
            b.write_dataset(self.data, self.lock, c)
            b.verify(a)
            self.assertEqual(b.load_dataset(a), b.load_dataset(a / "dataset.json.gz"))
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (c / path.name).read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                b.write_dataset(self.data, self.lock, a)
            raw = (a / "dataset.json").read_bytes()
            self.assertEqual(gzip.decompress((a / "dataset.json.gz").read_bytes()), raw)
            (a / "dataset.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                b.verify(a)

    def test_downloader_pin_cache_and_offline_behavior(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "Offline source missing"):
                b.acquire(self.lock, root, offline=True)
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                lang = Path(kwargs["filename"]).stem
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.raw[lang])
                return str(target)

            module = types.SimpleNamespace(hf_hub_download=download)
            with patch.dict("sys.modules", {"huggingface_hub": module}):
                self.assertEqual(b.acquire(self.lock, root), self.raw)
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertEqual(call["revision"], self.lock["revision"])
                self.assertEqual(call["repo_type"], "dataset")
                self.assertIs(call["token"], False)
            self.assertEqual(b.acquire(self.lock, root, offline=True), self.raw)
            path = Path(root) / self.lock["revision"] / "data/eng_Latn.jsonl"
            path.write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                b.acquire(self.lock, root, offline=True)

    def test_export_and_unknown_languages(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "mixed"
            b.export_beir(self.data, output, "eng_Latn", "*")
            rows = [json.loads(r) for r in (output / "corpus.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len((output / "qrels/test.tsv").read_text().splitlines()), 5)
            self.assertIn("source", b.read_json(output / "SOURCE.json"))
        with self.assertRaisesRegex(ValueError, "Unknown query language"):
            b.task(self.data, "deu_Latn")


if __name__ == "__main__":
    unittest.main()
