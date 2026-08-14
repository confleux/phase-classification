# AGENTS.md

## Status

Fresh `uv init` scaffold (2026-08). No functionality, tests, deps, or CI yet. Expect churn; verify anything this file says before trusting it.

## Toolchain

- Managed with **uv**, Python **3.14** (`/Users/.../phase-classification/pyproject.toml`, `.python-version`).
- Default branch is `dev` (not `main`).
- No `uv.lock` committed and no `[build-system]`/`[dependency-groups]` in `pyproject.toml` yet — this is still the boilerplate `uv init` state.

## Working here

- Run code with `uv run python main.py` (no entrypoint/script configured).
- `pyproject.toml` still has the placeholder `description = "Add your description here"`.
- No test, lint, or format tooling is configured; don't assume one exists.

## Python, UV & Git Workflow Rules
- **Environment & Execution**: Always run Python files using `uv run <script_path>`. Never use raw `python` or activate the environment manually.
- **Dependency Management**: To add or remove libraries, use `uv add <package>` or `uv remove <package>`. Never use `pip install`.
- **Testing & Tooling**: Run tests via `uv run pytest`. Run linting via `uv run ruff check .`.
- **Git Commit Compliance**: 
  - Work in highly isolated micro-steps.
  - Modify a single logical module, class, or function at a time.
  - Pause, execute verification tests, and explicitly ask the user for feedback before touching a second file.
  - Ensure all updates leave the `pyproject.toml` or `uv.lock` synchronized.
