#!/usr/bin/env bun
/**
 * Validate a built dataset directory: ./datasets/<dataset_name>/
 *
 * a) Structural schema check of dataset.json against
 *    "tiny-multilingual-retrieval.v1" (see src/types.ts for the contract
 *    and the reasoning behind the schema design).
 * b) Alignment check of MANIFEST.json against the actual dataset.json data
 *    (counts, sampling, schema, byte limit, file bytes + SHA-256).
 *
 * Usage: bun run validate [dataset_name]   (default: tiny-embedding-bench-v1)
 * Exit code 0 = all checks pass, 1 = at least one failure, 2 = bad invocation.
 *
 * WHY this script exists
 * ----------------------
 * The dataset is a frozen regression benchmark: results are only comparable
 * if everyone evaluates the exact same bytes. The builder already pins
 * upstream sources and a release fingerprint, but once the folder is copied
 * around (checkouts, archives, CI artifacts) nothing guarantees it arrived
 * intact. This script is the cheap local gate: re-check the schema and prove
 * the folder still matches its MANIFEST.json before trusting any score on it.
 *
 * WHY hand-rolled checks instead of a schema library (zod, JSON Schema, …)
 * ------------------------------------------------------------------------
 * The TypeScript types in src/types.ts are erased at runtime and cannot
 * validate anything; a schema dependency would be the only runtime
 * dependency in an otherwise dependency-free tool. Explicit checks keep the
 * error messages specific (they say WHICH record and WHICH field broke) and
 * run on a bare Bun install.
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { argv, exit } from "node:process";
import type { BuildManifest, TinyRetrievalDataset } from "./types";

/**
 * Collects failures instead of throwing on the first one: a corrupted
 * dataset usually breaks several checks, and seeing all of them at once
 * distinguishes "one bad file" from "systematically wrong build".
 */
class Report {
  errors = 0;
  ok(message: string): void {
    console.log(`  ✓ ${message}`);
  }
  fail(message: string): void {
    this.errors += 1;
    console.error(`  ✗ ${message}`);
  }
  check(condition: boolean, message: string): boolean {
    if (condition) this.ok(message);
    else this.fail(message);
    return condition;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sha256Hex(bytes: Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseJsonFile(path: string): unknown {
  return JSON.parse(readFileSync(path, "utf8"));
}

function validateDatasetShape(
  raw: unknown,
  datasetName: string,
  report: Report,
): TinyRetrievalDataset | null {
  console.log("\n[a] dataset.json schema checks");

  if (!isRecord(raw)) {
    report.fail("top-level value is an object");
    return null;
  }
  report.ok("top-level value is an object");
  const d = raw;

  report.check(d.schema === "tiny-multilingual-retrieval.v1", `schema is "tiny-multilingual-retrieval.v1" (got ${JSON.stringify(d.schema)})`);
  // The internal name is baked into the release-locked SHA-256 of
  // dataset.json, so it intentionally stays "belebele-tiny-20-v1" even when
  // the folder is named differently. Equality with the folder would be a
  // false failure — report it as information instead.
  report.check(typeof d.name === "string" && d.name.length > 0, `name is a non-empty string (got ${JSON.stringify(d.name)})`);
  if (d.name !== datasetName) {
    console.log(`  ℹ dataset name "${String(d.name)}" differs from folder name "${datasetName}" (release-locked, informational only)`);
  }

  report.check(isRecord(d.source), "source is an object");
  if (isRecord(d.source)) {
    for (const key of ["repo_id", "revision", "split", "url", "license", "license_url", "attribution", "citation", "changes"]) {
      report.check(typeof d.source[key] === "string", `source.${key} is a string`);
    }
  }

  // languages defines the positional contract for every text[] array, so it
  // must be validated (and its length trusted) before any record is checked.
  const languages = d.languages;
  const langCount = Array.isArray(languages) ? languages.length : 0;
  const languagesValid =
    report.check(Array.isArray(languages) && languages.length > 0, "languages is a non-empty array") &&
    report.check(
      (languages as unknown[]).every((l) => typeof l === "string" && l.length > 0),
      "every language is a non-empty string",
    ) &&
    report.check(new Set(languages as unknown[]).size === langCount, "language codes are unique");
  if (!languagesValid) return null;

  const sampling = d.sampling;
  report.check(isRecord(sampling), "sampling is an object");
  if (isRecord(sampling)) {
    for (const key of ["seed", "passages_per_language", "queries_per_language"]) {
      report.check(Number.isInteger(sampling[key]), `sampling.${key} is an integer`);
    }
    for (const key of ["method", "query_review", "allowlist_sha256"]) {
      report.check(typeof sampling[key] === "string", `sampling.${key} is a string`);
    }
  }

  const corpus = d.corpus;
  const queries = d.queries;
  const qrels = d.qrels;
  if (!report.check(Array.isArray(corpus) && corpus.length > 0, "corpus is a non-empty array")) return null;
  if (!report.check(Array.isArray(queries) && queries.length > 0, "queries is a non-empty array")) return null;
  if (!report.check(Array.isArray(qrels), "qrels is an array")) return null;

  const corpusIds = new Set<string>();
  let corpusBad = 0;
  (corpus as unknown[]).forEach((entry, i) => {
    const where = `corpus[${i}]`;
    if (!isRecord(entry) || typeof entry.id !== "string" || entry.id.length === 0) {
      report.fail(`${where} is an object with a non-empty string id`);
      corpusBad += 1;
      return;
    }
    // Duplicate ids would silently alias two different passages under one
    // qrel target, corrupting every metric computed on the dataset.
    if (corpusIds.has(entry.id)) {
      report.fail(`${where}.id ${entry.id} is duplicated`);
      corpusBad += 1;
    }
    corpusIds.add(entry.id);
    if (typeof entry.source_url !== "string" || !entry.source_url.startsWith("http")) {
      report.fail(`${where}.source_url is an http(s) URL string`);
      corpusBad += 1;
    }
    if (
      !Array.isArray(entry.text) ||
      entry.text.length !== langCount ||
      !entry.text.every((t) => typeof t === "string" && t.length > 0)
    ) {
      report.fail(`${where}.text has exactly ${langCount} non-empty strings`);
      corpusBad += 1;
    }
  });
  report.check(corpusBad === 0, `all ${(corpus as unknown[]).length} corpus entries valid (id, source_url, text[${langCount}])`);

  const queryIds = new Set<string>();
  let queriesBad = 0;
  (queries as unknown[]).forEach((entry, i) => {
    const where = `queries[${i}]`;
    if (!isRecord(entry) || typeof entry.id !== "string" || entry.id.length === 0) {
      report.fail(`${where} is an object with a non-empty string id`);
      queriesBad += 1;
      return;
    }
    if (queryIds.has(entry.id)) {
      report.fail(`${where}.id ${entry.id} is duplicated`);
      queriesBad += 1;
    }
    queryIds.add(entry.id);
    if (!Number.isInteger(entry.question_number) || (entry.question_number as number) < 1) {
      report.fail(`${where}.question_number is a positive integer`);
      queriesBad += 1;
    }
    if (
      !Array.isArray(entry.text) ||
      entry.text.length !== langCount ||
      !entry.text.every((t) => typeof t === "string" && t.length > 0)
    ) {
      report.fail(`${where}.text has exactly ${langCount} non-empty strings`);
      queriesBad += 1;
    }
  });
  report.check(queriesBad === 0, `all ${(queries as unknown[]).length} query entries valid (id, question_number, text[${langCount}])`);

  // Referential integrity: a qrel pointing at a missing query/passage would
  // drop or invent a gold label, skewing scores without any visible error.
  const qrelsPerQuery = new Map<string, number>();
  let qrelsBad = 0;
  (qrels as unknown[]).forEach((entry, i) => {
    const where = `qrels[${i}]`;
    if (!Array.isArray(entry) || entry.length !== 3) {
      report.fail(`${where} is a [query_id, passage_id, score] triple`);
      qrelsBad += 1;
      return;
    }
    const [qid, pid, score] = entry as [unknown, unknown, unknown];
    if (typeof qid !== "string" || !queryIds.has(qid)) {
      report.fail(`${where}[0] references an existing query id`);
      qrelsBad += 1;
    } else {
      qrelsPerQuery.set(qid, (qrelsPerQuery.get(qid) ?? 0) + 1);
    }
    if (typeof pid !== "string" || !corpusIds.has(pid)) {
      report.fail(`${where}[1] references an existing passage id`);
      qrelsBad += 1;
    }
    if (score !== 1) {
      report.fail(`${where}[2] is binary relevance 1`);
      qrelsBad += 1;
    }
  });
  report.check(qrelsBad === 0, "all qrels are valid [query_id, passage_id, 1] with resolvable references");
  // Exactly one positive per query is a property the evaluation math relies
  // on (Recall@k == Hit@k, Recall@100 == 1). Zero qrels strands a query;
  // two would silently change the ideal ranking.
  report.check(
    queryIds.size === qrelsPerQuery.size && [...qrelsPerQuery.values()].every((n) => n === 1),
    `every query has exactly one qrel (${qrelsPerQuery.size} qrels for ${queryIds.size} queries)`,
  );

  if (report.errors > 0) return null;
  return raw as unknown as TinyRetrievalDataset;
}

function validateManifest(
  manifestRaw: unknown,
  dataset: TinyRetrievalDataset,
  datasetDir: string,
  datasetBytes: Buffer,
  report: Report,
): void {
  console.log("\n[b] MANIFEST.json alignment checks");

  if (!isRecord(manifestRaw)) {
    report.fail("manifest is an object");
    return;
  }
  report.ok("manifest is an object");
  const m = manifestRaw;

  report.check(m.schema === "tiny-retrieval-build.v1", `manifest schema is "tiny-retrieval-build.v1" (got ${JSON.stringify(m.schema)})`);
  report.check(m.dataset_schema === dataset.schema, "manifest dataset_schema matches dataset.schema");

  // Deep string equality: the recipe recorded at build time must match the
  // recipe shipped inside the dataset, otherwise the two artifacts drifted
  // apart (e.g. one was rebuilt with a different seed or allowlist).
  report.check(
    JSON.stringify(m.sampling) === JSON.stringify(dataset.sampling),
    "manifest sampling recipe matches dataset.sampling",
  );

  const sourceLock = m.source_lock;
  report.check(isRecord(sourceLock), "manifest source_lock is an object");
  if (isRecord(sourceLock)) {
    report.check(sourceLock.repo_id === dataset.source.repo_id, "source_lock.repo_id matches dataset.source.repo_id");
    report.check(sourceLock.revision === dataset.source.revision, "source_lock.revision matches dataset.source.revision");
    report.check(sourceLock.split === dataset.source.split, "source_lock.split matches dataset.source.split");
    const lockCodes = Array.isArray(sourceLock.languages)
      ? (sourceLock.languages as unknown[]).map((l) => (isRecord(l) ? l.code : undefined))
      : [];
    // Order matters here: text[] arrays are positional, so a reordered
    // language list would mislabel every text in the dataset.
    report.check(
      JSON.stringify(lockCodes) === JSON.stringify(dataset.languages),
      "source_lock language codes match dataset.languages (same order)",
    );
  }

  const counts = m.counts;
  report.check(isRecord(counts), "manifest counts is an object");
  if (isRecord(counts)) {
    const langCount = dataset.languages.length;
    const expected = {
      languages: langCount,
      passage_groups: dataset.corpus.length,
      query_groups: dataset.queries.length,
      localized_passages: dataset.corpus.length * langCount,
      localized_queries: dataset.queries.length * langCount,
    } as const;
    for (const [key, actual] of Object.entries(expected)) {
      report.check(counts[key] === actual, `counts.${key} = ${actual} (manifest: ${JSON.stringify(counts[key])})`);
    }
  }

  // The <1 MB budget is a deliberate benchmark feature (see types.ts);
  // busting it means the artifact is no longer the frozen v1 release.
  const byteLimit = m.byte_limit;
  report.check(isRecord(byteLimit), "manifest byte_limit is an object");
  if (isRecord(byteLimit) && byteLimit.file === "dataset.json" && Number.isInteger(byteLimit.exclusive_max_bytes)) {
    report.check(
      datasetBytes.length < (byteLimit.exclusive_max_bytes as number),
      `dataset.json is ${datasetBytes.length} bytes < exclusive limit ${byteLimit.exclusive_max_bytes}`,
    );
  } else {
    report.fail("byte_limit describes dataset.json with an integer exclusive_max_bytes");
  }

  // Hash the raw bytes read from disk, never a re-serialization: JSON.parse
  // + JSON.stringify would normalize away exactly the byte-level differences
  // (whitespace, key order) that the manifest fingerprint is meant to catch.
  const files = m.files;
  report.check(isRecord(files), "manifest files is an object");
  if (isRecord(files)) {
    for (const [name, stat] of Object.entries(files)) {
      if (!isRecord(stat) || !Number.isInteger(stat.bytes) || typeof stat.sha256 !== "string") {
        report.fail(`files.${name} has integer bytes and string sha256`);
        continue;
      }
      const path = join(datasetDir, name);
      if (!existsSync(path)) {
        report.fail(`files.${name}: file is missing on disk`);
        continue;
      }
      const bytes = readFileSync(path);
      report.check(bytes.length === stat.bytes, `files.${name}.bytes = ${stat.bytes} (actual: ${bytes.length})`);
      report.check(sha256Hex(bytes) === stat.sha256, `files.${name}.sha256 matches actual file hash`);
    }
  }
}

function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

/**
 * Prints statistics MEASURED from the parsed dataset — nothing here is read
 * from the manifest's claims, so the numbers can be pasted into docs and
 * stay honest even if the manifest lies. (The manifest's own numbers are
 * cross-checked separately in validateManifest.)
 */
function printMeasuredStats(
  dataset: TinyRetrievalDataset,
  datasetBytes: Buffer,
  manifestRaw: unknown,
): void {
  const langCount = dataset.languages.length;

  // Language display names are metadata the dataset itself does not carry;
  // they come from the manifest's source_lock (validated above).
  const names = new Map<string, string>();
  if (isRecord(manifestRaw) && isRecord(manifestRaw.source_lock) && Array.isArray(manifestRaw.source_lock.languages)) {
    for (const l of manifestRaw.source_lock.languages) {
      if (isRecord(l) && typeof l.code === "string" && typeof l.name === "string") names.set(l.code, l.name);
    }
  }

  // Per-language UTF-8 footprint of the positional text[] arrays.
  const passageBytes = new Array<number>(langCount).fill(0);
  const queryBytes = new Array<number>(langCount).fill(0);
  for (const entry of dataset.corpus) {
    entry.text.forEach((t, i) => {
      passageBytes[i] = (passageBytes[i] ?? 0) + Buffer.byteLength(t, "utf8");
    });
  }
  for (const entry of dataset.queries) {
    entry.text.forEach((t, i) => {
      queryBytes[i] = (queryBytes[i] ?? 0) + Buffer.byteLength(t, "utf8");
    });
  }

  const passages = dataset.corpus.length;
  const queries = dataset.queries.length;
  const qrels = dataset.qrels.length;

  console.log("\nMeasured statistics (computed from dataset.json)\n");
  console.log("| Metric | Value |");
  console.log("|---|---:|");
  console.log(`| Languages | ${fmtInt(langCount)} |`);
  console.log(`| Passage groups (haystack size per language) | ${fmtInt(passages)} |`);
  console.log(`| Localized passages (total) | ${fmtInt(passages * langCount)} |`);
  console.log(`| Query groups (needles per language) | ${fmtInt(queries)} |`);
  console.log(`| Localized queries (total) | ${fmtInt(queries * langCount)} |`);
  console.log(`| Qrels | ${fmtInt(qrels)} |`);
  console.log(`| Positives per monolingual query | 1 of ${fmtInt(passages)} |`);
  console.log(`| Positives per mixed-language query | ${fmtInt(langCount)} of ${fmtInt(passages * langCount)} |`);
  console.log(`| dataset.json bytes | ${fmtInt(datasetBytes.length)} |`);
  console.log(`| dataset.json SHA-256 | \`${sha256Hex(datasetBytes)}\` |`);

  console.log("\n| # | Language | Code | Corpus bytes (UTF-8) | Query bytes (UTF-8) | Avg bytes / passage |");
  console.log("|---:|---|---|---:|---:|---:|");
  dataset.languages.forEach((code, i) => {
    const pb = passageBytes[i] ?? 0;
    const qb = queryBytes[i] ?? 0;
    console.log(
      `| ${i + 1} | ${names.get(code) ?? code} | \`${code}\` | ${fmtInt(pb)} | ${fmtInt(qb)} | ${fmtInt(Math.round(pb / passages))} |`,
    );
  });
}

function main(): void {
  // Flags start with "--"; the first positional argument is the dataset name.
  const args = argv.slice(2);
  const wantStats = args.includes("--stats");
  // Default matches the root Makefile's DATASET so the bare
  // `bun run validate` / `make validate` invocation just works.
  const datasetName = args.find((a) => !a.startsWith("-")) ?? "tiny-embedding-bench-v1";
  if (datasetName.includes("/") || datasetName.includes("..")) {
    // The name becomes a filesystem path below; refuse anything that could
    // escape ./datasets/ instead of validating an arbitrary directory.
    console.error("Usage: bun run validate [dataset_name] [--stats]   (default: tiny-embedding-bench-v1)");
    console.error("Validates ./datasets/<dataset_name>/{dataset.json,MANIFEST.json}");
    exit(2);
  }

  // Resolve relative to this file, not the cwd, so the script works from
  // any working directory.
  const datasetDir = join(import.meta.dir, "..", "datasets", datasetName);
  const datasetPath = join(datasetDir, "dataset.json");
  const manifestPath = join(datasetDir, "MANIFEST.json");

  const report = new Report();
  console.log(`Validating ${datasetDir}`);

  for (const path of [datasetPath, manifestPath]) {
    if (!existsSync(path)) {
      report.fail(`required file missing: ${path}`);
    }
  }
  if (report.errors > 0) exit(1);

  // Keep the raw dataset.json buffer around: the manifest check hashes the
  // exact on-disk bytes, which a parse/re-serialize round-trip would change.
  const datasetBytes = readFileSync(datasetPath);
  const dataset = validateDatasetShape(parseJsonFile(datasetPath), datasetName, report);
  const manifestRaw = parseJsonFile(manifestPath);
  if (dataset !== null) {
    validateManifest(manifestRaw, dataset, datasetDir, datasetBytes, report);
    if (wantStats) printMeasuredStats(dataset, datasetBytes, manifestRaw);
  }

  console.log(
    report.errors === 0
      ? `\nPASS: ${datasetName} is schema-valid and aligned with its MANIFEST.json`
      : `\nFAIL: ${report.errors} check(s) failed for ${datasetName}`,
  );
  exit(report.errors === 0 ? 0 : 1);
}

main();
