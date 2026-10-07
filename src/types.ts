/**
 * Canonical TypeScript view of the frozen benchmark dataset format
 * ("tiny-multilingual-retrieval.v1") produced by bench-distill-builder.
 *
 * WHY this module exists
 * ----------------------
 * `dataset.json` is consumed by JS/TS with a plain `JSON.parse` — there is no
 * decoder in the loop, so the compiler sees `any`. These interfaces are the
 * written contract between the builder and consumers. They cannot enforce
 * anything at runtime (types are erased), which is why src/validate.ts
 * re-checks the same contract against real bytes. Keep the two in sync.
 *
 * WHY the schema looks the way it does
 * ------------------------------------
 * - Positional `text[]` arrays instead of `{ [language]: text }` maps:
 *   repeating 20 language keys on every one of 2,050 records would cost a
 *   large fraction of the strict 1,000,000-byte budget the dataset is
 *   designed to fit in. Position i always means `languages[i]`, defined once
 *   at the top level. The arrays are ordered, never sparse.
 * - Shared ids across translations: all 20 translations of a passage are the
 *   SAME underlying text, so they share one `p-…` id (same for `q-…`). This
 *   lets qrels be stored once (50 triples instead of 1,000) and makes the
 *   cross-/mixed-language tasks expressible without id mapping tables.
 * - Binary qrels (`score` is always 1): the gold label is a *source-passage
 *   association* (this question was written about this passage), not a
 *   human-ranked relevance judgment. Other passages were never exhaustively
 *   judged, so there is no honest 0/2/3 scale to record.
 * - `source_url` on every passage: the data is CC-BY-SA-4.0 Wikimedia text;
 *   the link is the attribution/provenance trail and must be retained.
 */

export interface DatasetSource {
  /** Upstream Hugging Face repo the passages were sampled from. */
  repo_id: string;
  /**
   * Immutable upstream commit. Pinned (not a branch/tag) so the benchmark is
   * reproducible even if the upstream dataset is later modified or removed.
   */
  revision: string;
  split: string;
  url: string;
  license: string;
  license_url: string;
  attribution: string;
  citation: string;
  /** Human-readable diff vs. upstream: what sampling/screening changed. */
  changes: string;
}

export interface SamplingRecipe {
  /**
   * Seed for the SHA-256-based tie-breaking and ordering. Recorded so the
   * exact corpus/query ordering can be audited and reproduced.
   */
  seed: number;
  passages_per_language: number;
  queries_per_language: number;
  method: string;
  /**
   * Honesty marker: queries were screened by an assistant on English text
   * only; translations are inherited from the source, not natively reviewed.
   * Consumers must know this before treating scores as an evaluation standard.
   */
  query_review: string;
  /** Fingerprint of the screened English query allowlist used as input. */
  allowlist_sha256: string;
}

export interface CorpusEntry {
  /** Stable passage-group id shared by all translations, e.g. "p-6fb0c9d184ef5950". */
  id: string;
  /** Original Wikimedia source link. Metadata, not model input — retain for attribution. */
  source_url: string;
  /** Complete passage per language; length must equal `languages.length`. Never truncated. */
  text: string[];
}

export interface QueryEntry {
  /** Stable question-group id shared by all translations, e.g. "q-5d8579bfa8366f23". */
  id: string;
  /**
   * Original question number within its source passage. Kept because
   * (source_url, question_number) is the join key that aligned the upstream
   * per-language files, whose row orders differ.
   */
  question_number: number;
  /** Original question per language; length must equal `languages.length`. Original wording, never rewritten. */
  text: string[];
}

/**
 * [query_id, passage_id, relevance]. Relevance is binary and always 1
 * (see file header). Each query has exactly one qrel: one known positive
 * per monolingual query, so Recall@k == Hit@k.
 */
export type Qrel = [queryId: string, passageId: string, score: number];

export interface TinyRetrievalDataset {
  /**
   * Format version string. Bumped on any incompatible change so validators
   * and consumers reject silently-mutated files instead of misparsing them.
   */
  schema: string;
  /**
   * Release name ("belebele-tiny-20-v1"). Baked into the release-locked
   * SHA-256 of dataset.json, so it does NOT necessarily equal the folder
   * name the dataset is stored under (e.g. datasets/tiny-embedding-bench-v1).
   */
  name: string;
  source: DatasetSource;
  languages: string[];
  sampling: SamplingRecipe;
  corpus: CorpusEntry[];
  queries: QueryEntry[];
  qrels: Qrel[];
}

/** Byte size + content hash: enough to detect any tampering or truncated copy. */
export interface ManifestFileStat {
  bytes: number;
  sha256: string;
}

export interface ManifestCounts {
  languages: number;
  passage_groups: number;
  query_groups: number;
  /** passage_groups × languages — derived, kept so parsers can sanity-check cheaply. */
  localized_passages: number;
  /** query_groups × languages — derived, same reason. */
  localized_queries: number;
}

export interface SourceLockLanguage {
  code: string;
  name: string;
}

/**
 * The immutable upstream snapshot the dataset was built from: which commit,
 * which per-language files, and their exact bytes/hashes. The builder
 * re-verifies these pins on every run; the manifest carries them so an
 * auditor can check provenance without re-running the build.
 */
export interface SourceLock {
  repo_id: string;
  revision: string;
  split: string;
  source_rows_per_language: number;
  source_passages_per_language: number;
  languages: SourceLockLanguage[];
  files: Record<string, ManifestFileStat>;
}

/**
 * BuildManifest ("tiny-retrieval-build.v1") — the builder's signed summary of
 * what dataset.json SHOULD contain. The dataset is a frozen regression
 * benchmark: its value depends on everyone evaluating the exact same bytes.
 * The manifest is the independent statement of counts, recipe and file
 * hashes that src/validate.ts cross-checks the dataset against.
 */
export interface BuildManifest {
  schema: string;
  /** Value dataset.json must carry in its own `schema` field. */
  dataset_schema: string;
  source_lock: SourceLock;
  /** Must deep-equal dataset.sampling — guards against recipe drift. */
  sampling: SamplingRecipe;
  /**
   * The <1 MB constraint is a deliberate benchmark feature (the dataset is
   * meant to be trivially distributable and self-contained), so exceeding it
   * is a build failure, not a warning.
   */
  byte_limit: {
    file: string;
    exclusive_max_bytes: number;
  };
  counts: ManifestCounts;
  files: Record<string, ManifestFileStat>;
}
