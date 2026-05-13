# Project Overview

Medical imaging research project for Intracranial Hemorrhage (ICH) segmentation in PyTorch. Current work focuses on MAE self-supervised pre-training on the RSNA Intracranial Hemorrhage Detection dataset; downstream segmentation is planned.

## Linting and formatting

This project uses Ruff for both linting and formatting. Always invoke through `uv run`. When working with Ruff, invoke `/astral:ruff` to follow Astral's recommended usage.

- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Configuration lives in `pyproject.toml` under `[tool.ruff]`.
