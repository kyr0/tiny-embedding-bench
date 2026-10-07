# Build recipe for the frozen Belebele subset. The root Makefile delegates
# here; everything below assumes it runs inside this directory.
#
# WHY variables are overridable (?=): CI and offline environments need to
# swap the interpreter, venv location, output path and cache without editing
# the recipe, e.g. `make run PYTHON=python3.13 CACHE=/shared/cache`.

PYTHON ?= python3
VENV ?= .venv
PY = $(VENV)/bin/python
# Default output name matches the repository-level dataset folder
# (datasets/tiny-embedding-bench-v1) so builder-local and root-level builds
# land in the same place relative to their root.
OUTPUT ?= datasets/tiny-embedding-bench-v1
# Persistent download cache: upstream files are hash-pinned and large
# (~35 MB), so keeping them across runs makes rebuilds offline-friendly.
CACHE ?= .cache/belebele

.PHONY: setup run verify test offline
setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt
run:
	$(PY) build_dataset.py build --output "$(OUTPUT)" --cache-dir "$(CACHE)"
# Same recipe without network: fails unless every pinned file is already
# cached. Useful for reproducibility checks in sandboxes/CI.
offline:
	$(PY) build_dataset.py build --output "$(OUTPUT)" --cache-dir "$(CACHE)" --offline
verify:
	$(PY) build_dataset.py verify "$(OUTPUT)"
test:
	$(PYTHON) -m unittest discover -s tests -v
