#!/usr/bin/env python3
"""Build a small, aligned Belebele retrieval subset; no models or remote code."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SCHEMA = "tiny-multilingual-retrieval.v1"
SEED = 20261007
PASSAGES = 100
QUERIES = 50
MAX_BYTES = 1_000_000  # Strictly less than one decimal MB, not one MiB.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def passage_id(url):
    return "p-" + sha(url.encode("utf-8"))[:16]


def query_id(key):
    return "q-" + sha(packed(list(key)))[:16]


def priority(kind, key):
    # Stable across Python versions, input iteration order and hash randomization.
    return sha(packed([SEED, kind, key]))


def check_source_bytes(data, expected, label):
    require(len(data) == expected["bytes"], f"Source size mismatch: {label}")
    require(sha(data) == expected["sha256"], f"Source SHA-256 mismatch: {label}")


def acquire(lock, cache, offline=False):
    """Fetch only 20 pinned JSONL files; verify cached files on every invocation."""
    snapshot = Path(cache) / lock["revision"]

    def one(language):
        filename = f"data/{language}.jsonl"
        path = snapshot / filename
        if not path.exists():
            require(not offline, f"Offline source missing: {path}")
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ValueError("Install dependencies first: make setup") from exc
            try:
                path = Path(hf_hub_download(
                    repo_id=lock["repo_id"], repo_type="dataset", filename=filename,
                    revision=lock["revision"], local_dir=str(snapshot), token=False,
                ))
            except Exception as exc:
                raise ValueError(f"Download failed for {filename}: {type(exc).__name__}: {exc}") from exc
        raw = path.read_bytes()
        check_source_bytes(raw, lock["files"][filename], filename)
        print(f"Verified {language}: {len(raw):,} source bytes", file=sys.stderr)
        return language, raw

    languages = [entry["code"] for entry in lock["languages"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(one, languages))


def index_sources(raw_sources, lock):
    """Join by the original (link, question_number), never by row position."""
    indexed = {}
    languages = [entry["code"] for entry in lock["languages"]]
    require(set(raw_sources) == set(languages), "Source language set mismatch")
    for language in languages:
        raw = raw_sources[language]
        check_source_bytes(raw, lock["files"][f"data/{language}.jsonl"], language)
        questions, passages = {}, {}
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            require(bool(line), f"Blank source row: {language}:{line_number}")
            row = json.loads(line)
            require(row.get("dialect") == language, f"Wrong dialect: {language}:{line_number}")
            for name in ("link", "flores_passage", "question"):
                require(isinstance(row.get(name), str) and row[name].strip(),
                        f"Invalid {name}: {language}:{line_number}")
            number = row.get("question_number")
            require(type(number) is int and number > 0, "Invalid source question number")
            url, text = row["link"], row["flores_passage"]
            key = (url, number)
            require(key not in questions, f"Duplicate source question: {language} {key}")
            require(url not in passages or passages[url] == text,
                    f"Inconsistent passage for URL: {language} {url}")
            questions[key] = row
            passages[url] = text
        require(len(questions) == lock["source_rows_per_language"],
                f"Source row quota mismatch: {language}")
        require(len(passages) == lock["source_passages_per_language"],
                f"Source passage quota mismatch: {language}")
        require(len(set(passages.values())) == len(passages),
                f"Duplicate passage text under different URLs: {language}")
        indexed[language] = {"questions": questions, "passages": passages}
    reference = set(indexed[languages[0]]["questions"])
    for language in languages:
        require(set(indexed[language]["questions"]) == reference,
                f"Cross-language question-key mismatch: {language}")
    return indexed


def select(indexed, lock, allowlist, passage_count=PASSAGES, query_count=QUERIES):
    require(0 < query_count <= passage_count, "Require 0 < queries <= passages")
    require(allowlist["source_revision"] == lock["revision"], "Curation revision mismatch")
    languages = [entry["code"] for entry in lock["languages"]]
    reference = indexed["eng_Latn"]
    approved = set()
    for entry in allowlist["approved"]:
        key = (entry["source_url"], entry["question_number"])
        require(key not in approved, f"Duplicate curation entry: {key}")
        require(key in reference["questions"], f"Unknown curation entry: {key}")
        require(reference["questions"][key]["question"] == entry["english"],
                f"Curation text changed: {key}")
        approved.add(key)

    def passage_cost(url):
        return sum(len(indexed[l]["passages"][url].encode("utf-8")) for l in languages)

    def question_cost(key):
        return sum(len(indexed[l]["questions"][key]["question"].encode("utf-8"))
                   for l in languages)

    # One question per underlying passage: no pseudo-replication within a language.
    best = {}
    for key in sorted(approved, key=lambda k: (question_cost(k), priority("query", k))):
        best.setdefault(key[0], key)
    require(len(best) >= query_count,
            f"Only {len(best)} screened passage groups; need {query_count}")
    positive_urls = sorted(best, key=lambda u: (
        passage_cost(u) + question_cost(best[u]), priority("positive", u)))[:query_count]
    selected = set(positive_urls)
    fillers = sorted(set(reference["passages"]) - selected,
                     key=lambda u: (passage_cost(u), priority("filler", u)))
    require(len(selected) + len(fillers) >= passage_count, "Insufficient passage supply")
    selected.update(fillers[:passage_count - query_count])
    urls = sorted(selected, key=lambda u: priority("corpus-order", u))
    keys = sorted((best[u] for u in positive_urls), key=lambda k: priority("query-order", k))
    return urls, keys


def make_dataset(indexed, lock, allowlist, passage_count=PASSAGES, query_count=QUERIES):
    languages = [entry["code"] for entry in lock["languages"]]
    urls, keys = select(indexed, lock, allowlist, passage_count, query_count)
    dataset = {
        "schema": SCHEMA,
        "name": "belebele-tiny-20-v1",
        "source": {
            "repo_id": lock["repo_id"], "revision": lock["revision"], "split": "test",
            "url": "https://huggingface.co/datasets/facebook/belebele/tree/" + lock["revision"],
            "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "attribution": "Belebele: Lucas Bandarkar et al. (ACL 2024); Meta; underlying Wikimedia contributors.",
            "citation": "https://aclanthology.org/2024.acl-long.44/",
            "changes": "Short-passage subset; assistant-screened English questions; original translations and text preserved; binary question-to-source-passage labels.",
        },
        "languages": languages,
        "sampling": {"seed": SEED, "passages_per_language": passage_count,
                     "queries_per_language": query_count, "method": "screened-short-v1",
                     "query_review": "assistant-English-only; translations inherited, not independently reviewed",
                     "allowlist_sha256": sha(packed(allowlist))},
        "corpus": [{"id": passage_id(u), "source_url": u,
                    "text": [indexed[l]["passages"][u] for l in languages]} for u in urls],
        "queries": [{"id": query_id(k), "question_number": k[1],
                     "text": [indexed[l]["questions"][k]["question"] for l in languages]} for k in keys],
        "qrels": [[query_id(k), passage_id(k[0]), 1] for k in keys],
    }
    validate_dataset(dataset)
    return dataset


def validate_dataset(data):
    require(data.get("schema") == SCHEMA, "Unsupported dataset schema")
    languages = data["languages"]
    require(isinstance(languages, list) and languages and
            all(isinstance(l, str) and l for l in languages) and
            len(set(languages)) == len(languages), "Invalid/duplicate language IDs")
    n = len(languages)
    id_sets = {}
    for field, quota in (("corpus", "passages_per_language"), ("queries", "queries_per_language")):
        records = data[field]
        expected = data["sampling"][quota]
        require(type(expected) is int and expected > 0, "Invalid quota")
        require(isinstance(records, list) and len(records) == expected, f"{field} quota mismatch")
        seen = set()
        for row in records:
            require(isinstance(row["id"], str) and row["id"] and row["id"] not in seen,
                    f"Invalid/duplicate {field} ID")
            seen.add(row["id"])
            require(isinstance(row["text"], list) and len(row["text"]) == n,
                    f"{field} text/language count mismatch")
            require(all(isinstance(t, str) and t.strip() for t in row["text"]),
                    f"Empty/non-string {field} text")
            if field == "corpus":
                require(isinstance(row["source_url"], str) and
                        row["id"] == passage_id(row["source_url"]), "Passage ID/source mismatch")
            else:
                require(type(row["question_number"]) is int and row["question_number"] > 0,
                        "Invalid query source number")
        for i in range(n):
            require(len({r["text"][i] for r in records}) == len(records),
                    f"Duplicate {field} text: {languages[i]}")
        id_sets[field] = seen
    corpus = {r["id"]: r for r in data["corpus"]}
    queries = {r["id"]: r for r in data["queries"]}
    judged, positives = set(), set()
    require(len(data["qrels"]) == len(queries), "Exactly one qrel per query required")
    for relation in data["qrels"]:
        require(isinstance(relation, list) and len(relation) == 3, "Invalid qrel shape")
        q, p, relevance = relation
        require(q in queries and p in corpus, "Dangling qrel")
        require(q not in judged and p not in positives, "Repeated query or positive passage")
        require(type(relevance) is int and relevance == 1, "Binary positive relevance required")
        require(q == query_id((corpus[p]["source_url"], queries[q]["question_number"])),
                "Query ID/qrel/source mismatch")
        judged.add(q)
        positives.add(p)
    require(judged == id_sets["queries"], "Missing query judgment")


def load_dataset(path):
    path = Path(path)
    if path.is_dir():
        path = path / "dataset.json"
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    validate_dataset(data)
    return data


def check_release(data):
    """A v1 rebuild must reproduce the shipped benchmark, not just any valid subset."""
    expected = read_json(ROOT / "release-lock.json")
    check_source_bytes(packed(data) + b"\n", expected["dataset.json"], "frozen v1 dataset")


def task(data, query_language, corpus_language=None):
    """Return BEIR-shaped corpus/queries/qrels; '*' includes ALL translations."""
    validate_dataset(data)
    languages = data["languages"]
    corpus_language = corpus_language or query_language
    require(query_language in languages, f"Unknown query language: {query_language}")
    require(corpus_language == "*" or corpus_language in languages,
            f"Unknown corpus language: {corpus_language}")
    targets = languages if corpus_language == "*" else [corpus_language]
    corpus = {f"{l}:{r['id']}": {"title": "", "text": r["text"][languages.index(l)]}
              for l in targets for r in data["corpus"]}
    queries = {f"{query_language}:{r['id']}": r["text"][languages.index(query_language)]
               for r in data["queries"]}
    qrels = {f"{query_language}:{q}": {f"{l}:{p}": rel for l in targets}
             for q, p, rel in data["qrels"]}
    return corpus, queries, qrels


def gzip_bytes(raw):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as out:
        out.write(raw)
    return buffer.getvalue()


def review_html(data):
    i = data["languages"].index("eng_Latn")
    corpus = {r["id"]: r for r in data["corpus"]}
    positive = {q: p for q, p, _ in data["qrels"]}
    sections = []
    for row in data["queries"]:
        passage = corpus[positive[row["id"]]]
        sections.append("<article><h2>" + html.escape(row["text"][i]) + "</h2><p>" +
                        html.escape(passage["text"][i]) + "</p><small>" +
                        html.escape(row["id"] + " | " + passage["source_url"]) + "</small></article>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Belebele query review</title>"
            "<style>body{font:17px/1.5 system-ui;max-width:900px;margin:32px auto;padding:16px}"
            "article{border-top:1px solid #aaa;padding:20px 0}h2{font-size:19px}small{overflow-wrap:anywhere}</style>"
            "<h1>English query screen</h1><p>Assistant-screened originals. Other translations have not "
            "been independently reviewed. Text is retained verbatim; historic statements are source content.</p>" +
            "".join(sections) + "</html>").encode("utf-8")


def atomic_directory(output, files):
    output = Path(output)
    require(not output.exists(), f"Output already exists; choose a new path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    try:
        for name, content in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        require(not output.exists(), f"Output appeared during build: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_dataset(data, lock, output, max_bytes=MAX_BYTES):
    validate_dataset(data)
    raw = packed(data) + b"\n"
    require(len(raw) < max_bytes,
            f"Dataset is {len(raw):,} bytes; requires < {max_bytes:,}. No text or quota was reduced.")
    files = {"dataset.json": raw, "dataset.json.gz": gzip_bytes(raw),
             "review.html": review_html(data)}
    manifest = {
        "schema": "tiny-retrieval-build.v1", "dataset_schema": SCHEMA,
        "source_lock": lock, "sampling": data["sampling"],
        "byte_limit": {"file": "dataset.json", "exclusive_max_bytes": max_bytes},
        "counts": {"languages": len(data["languages"]), "passage_groups": len(data["corpus"]),
                   "query_groups": len(data["queries"]),
                   "localized_passages": len(data["languages"]) * len(data["corpus"]),
                   "localized_queries": len(data["languages"]) * len(data["queries"])},
        "files": {name: {"bytes": len(content), "sha256": sha(content)}
                  for name, content in files.items()},
    }
    files["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    atomic_directory(output, files)
    return manifest


def verify(output):
    output = Path(output)
    manifest = read_json(output / "MANIFEST.json")
    require(manifest.get("schema") == "tiny-retrieval-build.v1", "Unknown manifest schema")
    require(set(manifest["files"]) == {"dataset.json", "dataset.json.gz", "review.html"},
            "Manifest file set mismatch")
    for name, expected in manifest["files"].items():
        check_source_bytes((output / name).read_bytes(), expected, name)
    raw = (output / "dataset.json").read_bytes()
    require(len(raw) < manifest["byte_limit"]["exclusive_max_bytes"], "Byte cap violated")
    require(gzip.decompress((output / "dataset.json.gz").read_bytes()) == raw, "Gzip/raw mismatch")
    data = load_dataset(output)
    require(raw == packed(data) + b"\n", "Noncanonical dataset serialization")
    require(data["sampling"] == manifest["sampling"], "Sampling/manifest mismatch")
    require(data["source"]["revision"] == manifest["source_lock"]["revision"], "Revision mismatch")
    require(data["languages"] == [r["code"] for r in manifest["source_lock"]["languages"]],
            "Language order mismatch")
    expected_counts = {"languages": len(data["languages"]), "passage_groups": len(data["corpus"]),
                       "query_groups": len(data["queries"]),
                       "localized_passages": len(data["languages"]) * len(data["corpus"]),
                       "localized_queries": len(data["languages"]) * len(data["queries"])}
    require(manifest["counts"] == expected_counts, "Manifest count mismatch")
    return manifest


def export_beir(data, output, query_language, corpus_language=None):
    corpus, queries, qrels = task(data, query_language, corpus_language)
    tsv = io.StringIO(newline="")
    writer = csv.writer(tsv, delimiter="\t", lineterminator="\n")
    writer.writerow(["query-id", "corpus-id", "score"])
    writer.writerows((q, p, score) for q, rels in qrels.items() for p, score in rels.items())
    files = {
        "corpus.jsonl": b"".join(packed({"_id": key, **value}) + b"\n" for key, value in corpus.items()),
        "queries.jsonl": b"".join(packed({"_id": key, "text": text}) + b"\n" for key, text in queries.items()),
        "qrels/test.tsv": tsv.getvalue().encode("utf-8"),
        "SOURCE.json": packed({"source": data["source"], "sampling": data["sampling"],
                              "query_language": query_language,
                              "corpus_language": corpus_language or query_language,
                              "passage_sources": {r["id"]: r["source_url"] for r in data["corpus"]}}) + b"\n",
    }
    atomic_directory(output, files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Download, screen, select, validate and write the fixed profile")
    build.add_argument("--output", type=Path, default=Path("datasets/belebele-tiny-20-v1"))
    build.add_argument("--cache-dir", type=Path, default=Path(".cache/belebele"))
    build.add_argument("--offline", action="store_true", help="Use only hash-verified cached source files")
    check = sub.add_parser("verify", help="Check schema, quotas, references, byte cap and artifact hashes")
    check.add_argument("dataset", type=Path)
    export = sub.add_parser("export", help="Write a monolingual, cross-language or mixed BEIR-format task")
    export.add_argument("dataset", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--query-language", default="eng_Latn")
    export.add_argument("--corpus-language", help="Defaults to query language; '*' includes all languages")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            require(not args.output.exists(), f"Output already exists; choose a new path: {args.output}")
            lock = read_json(ROOT / "source-lock.json")
            allowlist = read_json(ROOT / "query-allowlist.json")
            indexed = index_sources(acquire(lock, args.cache_dir, args.offline), lock)
            data = make_dataset(indexed, lock, allowlist)
            check_release(data)
            manifest = write_dataset(data, lock, args.output)
            verify(args.output)
        elif args.command == "verify":
            manifest = verify(args.dataset)
            check_release(load_dataset(args.dataset))
        else:
            data = load_dataset(args.dataset)
            check_release(data)
            export_beir(data, args.output, args.query_language, args.corpus_language)
            print(f"Exported {args.output}")
            return 0
        print(json.dumps({"counts": manifest["counts"], "files": manifest["files"]}, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
