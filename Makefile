.PHONY: install quality train serve

install:
	uv sync --frozen --extra dev

quality:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run mypy src
	uv run pytest

train:
	uv run faturaiz train

serve:
	uv run uvicorn faturaiz.api:app --host 127.0.0.1 --port 8000

