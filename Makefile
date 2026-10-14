.PHONY: install test lint run app

install:
	pip install -r requirements.txt pytest pytest-cov ruff
	pip install -e . --no-deps

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -q --cov=shiftcast

run:
	python -m shiftcast.pipeline --out outputs --assets assets

app:
	python -m shiftcast.app
