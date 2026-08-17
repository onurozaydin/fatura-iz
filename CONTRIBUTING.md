# Contributing

Use a focused branch and include tests for behavior changes. Before opening a pull request, run:

```bash
uv sync --frozen --extra dev
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
uv run faturaiz train
```

Never commit real invoices, personal data, credentials, or unlicensed datasets. New synthetic
scenarios must be documented and deterministically generated.
