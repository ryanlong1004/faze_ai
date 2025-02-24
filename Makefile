
all: test sort lint
# Run tests
test:
	pytest -vv

# Run ruff for linting and formatting
lint:
	ruff check --fix-only && ruff format

sort:
	isort .

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

.PHONY: all clean test lint