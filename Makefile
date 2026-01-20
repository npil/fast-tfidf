.PHONY: mypy
mypy:
	poetry run mypy . --cache-fine-grained


.PHONY: lint
lint:
	poetry run ruff check
	poetry run ruff format --check
	make mypy


.PHONY: fix
fix:
	poetry run ruff format
	poetry run ruff check --fix


.PHONY: app
app:
	poetry run python -m fraud_detector.main


.PHONY: test
test:
	poetry run pytest


.PHONY: test-lint
test-lint:
	make test
	make lint


.PHONY: install
install:
	poetry install
