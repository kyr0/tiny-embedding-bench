# Tiny multilingual retrieval builder

Build a frozen **Belebele subset: 20 languages × 100 passages × 50 queries**, with binary source-passage relevance. One Python script; one download dependency; no model, GPU, remote dataset code, or annotation API required.

**Verified v1:** `dataset.json` is **954,985 bytes**; gzip is **438,478 bytes**. Every original text string is retained exactly. The strict `<1,000,000`-byte limit applies to the self-contained `dataset.json`. The gzip copy, review page, manifests and code are additional files. First acquisition downloads **35,019,937 bytes** of upstream JSONL, then caches it.

## Run

**From the repository root** — one line to create the virtualenv, install the download dependency and (re)create the dataset:

```bash
make setup create-dataset
```

This builds `datasets/tiny-embedding-bench-v1/` at the repository root, removing any existing copy first (the builder never overwrites). Upstream downloads are served from the builder cache when present. Afterwards, `make validate` at the root schema-checks `dataset.json` and verifies it against `MANIFEST.json` (requires [Bun](https://bun.sh)).

Python **3.12+**. The same steps can be run inside `bench-distill-builder/`:

```bash
make setup
make run
make verify
make test
```

Output: `datasets/tiny-embedding-bench-v1/`. The script downloads public Hugging Face files without login. It checks their pinned SHA-256 values on every run and joins translations by `(link, question_number)`, since row orders differ between languages.

The included **prebuilt dataset** is under `example/belebele-tiny-20-v1/`. Inspect `review.html` locally, or verify it without installing anything:

```bash
python3 build_dataset.py verify example/belebele-tiny-20-v1
```

Direct CLI and an offline rebuild:

```bash
.venv/bin/python build_dataset.py build --output datasets/my-copy
.venv/bin/python build_dataset.py build --offline --output datasets/offline-copy
```

Downloads use `.cache/belebele/<revision>/`; change it with `--cache-dir` or `make run CACHE=/path/to/cache`. Completed files survive interrupted downloads. Missing offline files, hash mismatches, invalid alignment, insufficient quotas, or exceeding the byte cap cause explicit failure. Existing output directories are never overwritten: choose a new `--output` or `make run OUTPUT=...`. Use a separate output path for concurrent runs.

For a SOCKS proxy environment, install `socksio` into the virtual environment if the downloader reports that it is missing.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Download, select, validate, write, verify and export |
| `source-lock.json` | Immutable HF commit and exact source-file sizes/SHA-256 |
| `query-allowlist.json` | 77 original English questions from 63 passage groups, screened by the assistant |
| `release-lock.json` | Expected final JSON size/hash; accidental benchmark changes fail |
| `dataset.json` | All 2,000 passages, 1,000 queries, IDs, source mapping and shared qrels |
| `dataset.json.gz` | Gzip copy with fixed timestamp and no embedded filename |
| `MANIFEST.json` | Source lock, sampling recipe, counts and output hashes |
| `review.html` | The 50 selected English query/passage pairs for inspection |

The last four files are produced in the output directory. No `datasets`, PyArrow, pandas or MTEB package is needed.

## Frozen sampling recipe

1. Read the pinned 900 original rows per language; require 488 unique source passages and identical `(source URL, question number)` sets across all languages. Validate per-language passage consistency and reject duplicates.
2. Use the bundled English query allowlist. It was screened from questions belonging to the **150 shortest passage groups**, measured by summed UTF-8 passage bytes across all 20 languages. A question needs an explicit subject or named entity and must be interpretable without omitted answer options or an unresolved passage reference. Original wording, including “According to the passage,” remains unchanged.
3. Within each approved source passage, choose the approved question with the smallest total UTF-8 size across languages. Choose **50 positive passage groups** by combined passage-plus-question bytes.
4. Add the **50 shortest remaining passage groups** as candidates without a selected query. Their lack of a qrel is not a human judgment that they are irrelevant to every question.
5. Break ties and order the corpus and queries using SHA-256 with seed **20261007**. Passage and query orders are independent. IDs hash source keys; translations share underlying IDs.
6. Write compact UTF-8 JSON, validate every qrel and count, require `<1,000,000` bytes including the final newline, and require the published v1 fingerprint. Publish the output directory only after all content is ready.

This is a deterministic, size-biased heuristic, not a uniform random sample or a proof of globally minimum size. No truncation, generated query rewriting, source substitution or silent count reduction occurs. Every selected question has a distinct positive passage; all languages use the same 100 underlying passages and 50 question identities.

The query review is **assistant screening of English only**. Translations are original aligned source questions; no independent native-speaker review or exhaustive relevance adjudication is claimed. Inspect the review page before treating this as an evaluation standard. Changing the source, screen or sampling requires a new benchmark version and release fingerprint.

## Languages and schema

Order: English, German, French, Spanish, Portuguese, Italian, Dutch, Polish, Russian, Turkish, Modern Standard Arabic, Hindi, Bengali, Urdu, Indonesian, Vietnamese, Thai, Simplified Chinese, Japanese, Korean. `source-lock.json` maps each name to its FLORES language code.

| Field | Meaning |
|---|---|
| `languages[i]` | Language code for every `text[i]` |
| `corpus[].id` | Stable passage-group ID |
| `corpus[].source_url` | Original Wikimedia source link; retain for provenance/attribution |
| `corpus[].text[i]` | Complete passage in language `languages[i]` |
| `queries[].id` | Stable question-group ID |
| `queries[].question_number` | Original question number within its source passage |
| `queries[].text[i]` | Original question in language `languages[i]` |
| `qrels[]` | `[query_id, passage_id, 1]`; shared across translations |
| `source`, `sampling` | Provenance, license, method, quotas, seed and screen fingerprint |

The small text arrays avoid repeating 20 language keys on every record. They are ordered, not sparse. Feed only the text to an embedding model; source URLs, qrels and provenance are metadata.

```python
from build_dataset import load_dataset, task

data = load_dataset("example/belebele-tiny-20-v1")
corpus, queries, qrels = task(data, "deu_Latn")
# corpus: {"deu_Latn:p-...": {"title": "", "text": "..."}, ...}
# queries: {"deu_Latn:q-...": "...", ...}
# qrels: {"deu_Latn:q-...": {"deu_Latn:p-...": 1}, ...}

cross = task(data, "eng_Latn", "deu_Latn")  # 100 German candidates
mixed = task(data, "eng_Latn", "*")        # 2,000 candidates; 20 positives/query
```

JavaScript consumption needs only `JSON.parse`:

```js
const dataset = await (await fetch("/dataset.json")).json();
const i = dataset.languages.indexOf("deu_Latn");
if (i < 0) throw new Error("Language unavailable");
const corpus = dataset.corpus.map(({ id, text }) => ({ id, text: text[i] }));
const queries = dataset.queries.map(({ id, text }) => ({ id, text: text[i] }));
const qrels = new Map(dataset.qrels.map(([qid, pid, score]) => [qid, { [pid]: score }]));
```

## Optional BEIR-format export

```bash
python3 build_dataset.py export example/belebele-tiny-20-v1 \
  --query-language deu_Latn --output exports/de-de

python3 build_dataset.py export example/belebele-tiny-20-v1 \
  --query-language eng_Latn --corpus-language deu_Latn --output exports/en-de

python3 build_dataset.py export example/belebele-tiny-20-v1 \
  --query-language eng_Latn --corpus-language '*' --output exports/en-all
```

Each export contains `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` and `SOURCE.json`. IDs are language-prefixed. The mixed corpus correctly labels **all 20 translations** of the positive passage as relevant. Expanded exports duplicate data and have no 1 MB guarantee.

## Evaluation interpretation

- Primary task: query language L → its 100 same-language passages. Report nDCG@10, MRR@10 and Hit@1/5/10; average per language, then equally across languages. This builder does not run an embedding model or calculate model scores.
- One known positive per monolingual query: `Recall@k = Hit@k`; `Recall@100 = 1`. Gold is a source-passage association, not a human-ranked list of all candidates. Other potentially relevant passages are not exhaustively judged.
- Evaluate lexical, dense and lexical+dense RRF on this identical frozen subset. Keep the test fixed; tune fusion settings elsewhere.
- Report cross-language and mixed-language tasks separately. In the mixed task there are 20 positives, so recall and hit rate differ and nDCG needs the correct multi-positive ideal ranking.
- This is a **custom retrieval regression subset**, not an official MTEB score. Short texts, 100 candidates and assistant curation bias its difficulty and may change model ordering.
- There are 50 underlying query-source groups reused across 20 languages; the 1,000 localized queries are not independent examples. For uncertainty, bootstrap passage groups jointly across translations.

The exact checks performed are the `verify` command, the unit suite (`make test`) and, at the repository root, `make validate`; see [NOTICE.md](NOTICE.md) for attribution and upstream references.
