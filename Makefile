.PHONY: mypy
mypy:
	poetry run mypy .


.PHONY: isort
isort:
	poetry run isort --check --diff .


.PHONY: lint
lint:
	poetry run ruff check
	make isort
	make mypy


.PHONY: fix
fix:
	poetry run isort .
	poetry run ruff format


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
