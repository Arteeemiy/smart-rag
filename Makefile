.PHONY: install run dev lint test build-index format

install:
	poetry install

run:
	poetry run python -m smart_rag

dev:
	poetry run uvicorn smart_rag.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	poetry run ruff check .
	poetry run ruff format --check .

test:
	poetry run pytest

build-index:
	poetry run python scripts/build_index.py

format:
	poetry run ruff format .
