# GitHub Publication Manifest

## 1. Final publication status

The repository passed the pre-publication audit and the listed hygiene fixes
have been applied:

* `.DS_Store` removed;
* `.gitignore` extended;
* AutoDock Vina binary excluded and documented;
* `requests` declared in `environment.yml`;
* `LICENSE` and `data/README.md` added;
* development-history docs archived under `docs/archive/`;
* README Quick Start and data-attribution link added;
* `CITATION.cff` added;
* publication manifest created.

No scientific results or model parameters were changed.

## 2. Files/directories to COMMIT

The full Group A list from `docs/github_prepublication_audit.md`, including:

```text
README.md
LICENSE
CITATION.cff
environment.yml
environment-docking.yml
.gitignore
src/
notebooks/
data/README.md
data/processed/
data/candidates/
data/docking/ (PDB, box, ligands; not the Vina binary)
docs/ (including docs/archive/)
results/figures/
selected final/support result CSV and JSON files
```

## 3. Patterns/files to IGNORE

```text
results/models/
data/raw/*
data/interim/*
data/docking/vina
__pycache__/
*.py[cod]
.DS_Store
.ipynb_checkpoints/
.env / .env.*
.pytest_cache/
.mypy_cache/
*.egg-info/
.idea/
.vscode/
*.log
*.tmp
```

## 4. Files intentionally retained locally but excluded

```text
results/models/          trained model artifacts (regenerate with scripts)
data/raw/                ChEMBL downloads (regenerate with download script)
data/interim/            preprocessing intermediates (regenerate)
data/docking/vina        platform-specific AutoDock Vina executable
```

## 5. Remaining manual-review items

* Confirm LICENSE author name before publishing.
* Confirm current ChEMBL and RCSB PDB terms/attribution for derived data.
* Decide whether to publish `results/m7_docking_out/*.pdbqt` and the
  superseded M6 shortlist files.
* Decide whether `docs/archive/` files should remain public or stay private.

## 6. Exact commands the user should run next

Run from the project root after reviewing the manifest:

```bash
git init
git status --short
```

Do not run `git add .` until the first `git status --short` output has been
inspected. Git was not initialized by this cleanup task.
