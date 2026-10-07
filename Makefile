# Root facade over the real build recipe in bench-distill-builder/Makefile.
# WHY delegate instead of duplicating: the builder owns the recipe and its
# invariants (pinned upstream hashes, release fingerprint); duplicating flags
# here would let the two drift apart. This file only adapts paths so the
# canonical workflow runs from the repository root.

BUILDER := bench-distill-builder
# Dataset folder name; src/validate.ts defaults to the same name, so
# `make create-dataset` + `make validate` compose with zero arguments.
DATASET := datasets/tiny-embedding-bench-v1

.PHONY: setup create-dataset validate

setup:
	$(MAKE) -C $(BUILDER) setup

# WHY rm -rf first: the builder deliberately refuses to overwrite an existing
# output directory (a stale or half-written dataset must never be silently
# reused), so "re-create" has to mean explicit removal followed by a rebuild.
# Safe to re-run: upstream files come from the builder's .cache when present.
# WHY $(CURDIR): make -C changes into $(BUILDER) before running, so a
# relative OUTPUT would land inside the builder directory instead of at root.
create-dataset:
	rm -rf "$(DATASET)"
	$(MAKE) -C $(BUILDER) run OUTPUT="$(CURDIR)/$(DATASET)"

validate:
	bun run validate $(notdir $(DATASET))

# Print statistics measured from dataset.json itself (never from the
# manifest's claims) — the numbers quoted in the root README come from here.
stats:
	bun run stats $(notdir $(DATASET))
