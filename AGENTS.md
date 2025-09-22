# Repository Guidelines

## Project Structure & Module Organization
Core training entry points (`train.py`, `inference.py`) live at the repository root and accept dataset or checkpoint paths via CLI flags. Shared neural layers and attention blocks sit under `models/` and `training/`, while reusable dataset helpers stay in `bucketing.py` and `cache.py`. VAE weights and configs reside in `vae/`; keep experiment outputs inside `runs/` so the git history stays clean. Place reusable inference pipelines in `inference/`, and mirror any new unit tests under a matching subtree inside `tests/`.

## Build, Test, and Development Commands
Use `UV_CACHE_DIR=.uv-cache uv sync` to install the locked Python 3.11 toolchain before executing any scripts so that dependencies stay inside the repo-local cache. Inspect training options with `UV_CACHE_DIR=.uv-cache uv run python train.py --help` and launch a short smoke run using dataset placeholders. Run inference sanity checks with `UV_CACHE_DIR=.uv-cache uv run python inference.py --config vae/config.json --image path/to/input.png`. Execute the validation suite with `UV_CACHE_DIR=.uv-cache uv run pytest` so the virtual environment stays consistent. For ad-hoc utilities, prefer `UV_CACHE_DIR=.uv-cache uv run python path/to/script.py` rather than invoking the system interpreter directly.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive snake_case for functions and variables, and PascalCase for classes. Type annotate public APIs and document non-obvious tensor shapes inline. Keep modules import-safe with `if __name__ == "__main__":` guards to avoid side effects during reuse. When touching shared utilities, add concise docstrings and maintain deterministic behavior.

## Testing Guidelines
Write tests with `pytest`, seeding random generators inside fixtures for reproducibility. Name cases `test_<module>_<behavior>` and mirror the source layout (e.g., `tests/training/test_scheduler.py`). Include smoke tests that execute a single mini-batch for new training loops to catch interface drift early. Run `uv run pytest -q` locally before pushing and ensure new datasets or checkpoints include validation steps in the PR description.

## Commit & Pull Request Guidelines
Use imperative, present-tense commit subjects (e.g., `Add aspect-ratio sampler`) and group related edits together. PR descriptions should summarize training or metric impact, list reproduction commands, and link tracking issues. Attach before/after artifacts—TensorBoard screenshots, sample renders, or diffusions—and call out outstanding TODOs. Note any secrets or dataset paths that reviewers need to recreate the run, and keep environment changes confined to `pyproject.toml` and `uv.lock`.

## Security & Configuration Tips
Store credentials in a local `.env` and load them explicitly; never commit API keys or dataset tokens. Reference checkpoint directories under `checkpoints/` or `runs/` using relative paths to support reproducible exports. When adding new configs, place them inside `configs/` and document expected environment variables in the file header for quick auditing.
