SHELL := /usr/bin/env bash -O globstar

.PHONY: check-licenses
check-licenses:
	poetry export > requirements.txt
	poetry run liccheck -l PARANOID

.PHONY: fix
fix:
	poetry run black .
	poetry run isort .

.PHONY: install
install:
	LDFLAGS="${LDFLAGS} -fno-lto -Wl,--no-as-needed" poetry install

.PHONY: lint
lint:
	poetry run black --check --diff tests fast_tfidf
	poetry run isort --check tests fast_tfidf
	poetry run flake8 tests fast_tfidf
	poetry run mypy tests fast_tfidf

.PHONY: test
test:
	poetry run pytest ./tests

.PHONY: lint-test
lint-test:
	$(MAKE) lint
	$(MAKE) test

.PHONY: publish
publish:
	poetry publish --build
