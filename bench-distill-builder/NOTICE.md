# Data attribution and sources

The included dataset and the original question text in `query-allowlist.json` derive from **Belebele**, by Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer and Madian Khabsa (Meta / paper authors), with underlying passage text from Wikimedia contributors.

The pinned [Hugging Face dataset card](https://huggingface.co/datasets/facebook/belebele/blob/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/README.md) declares **Creative Commons Attribution-ShareAlike 4.0 International**. The derivative data is distributed under the same license: [license and legal code](https://creativecommons.org/licenses/by-sa/4.0/). Preserve this attribution, the license, source links, source revision and indication of changes when redistributing the data. No upstream endorsement is implied.

Changes: selection of a small size-biased aligned subset; assistant screening of original English questions; omission of multiple-choice options and answer labels; deduplicated passage storage; compact JSON; binary source-passage retrieval qrels; optional retrieval task exports. The retained passage and question strings are not rewritten or truncated. The frozen JSONL SHA-256 values are in `source-lock.json`; original passage URLs remain in the dataset and exports.

This license statement concerns the upstream-derived data; it does not claim ownership of the upstream material.

Primary sources inspected on 2026-10-07:

- [Pinned dataset repository, revision 7899cdfa4e1e0d733fd77c848e2c273cb1d32be2](https://huggingface.co/datasets/facebook/belebele/tree/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2)
- [Belebele upstream repository](https://github.com/facebookresearch/belebele)
- [Paper, ACL 2024, pp. 749–775](https://aclanthology.org/2024.acl-long.44/)
- [MTEB BelebeleRetrieval implementation](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/multilingual/belebele_retrieval.py) — conceptual retrieval adaptation; no MTEB runtime or copied implementation is used.
- [Hugging Face download API documentation](https://huggingface.co/docs/huggingface_hub/guides/download) — exact revision and `hf_hub_download` use.

```bibtex
@inproceedings{bandarkar-etal-2024-belebele,
  title = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
  author = {Bandarkar, Lucas and Liang, Davis and Muller, Benjamin and Artetxe, Mikel and Shukla, Satya Narayan and Husa, Donald and Goyal, Naman and Krishnan, Abhinandan and Zettlemoyer, Luke and Khabsa, Madian},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2024},
  pages = {749--775},
  url = {https://aclanthology.org/2024.acl-long.44/}
}
```
