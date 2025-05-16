.PHONY: install test clean lint coverage coverage-html build publish

install:
	pip install -r requirements.txt

lint:
	isort src/ tests/
	python -m pylint src/ tests/

test:
	@export $(shell cat .env | xargs) && pytest --tb=short -v

coverage:
	pytest --cov=src --cov-report=term-missing

coverage-html:
	pytest --cov=src --cov-report=html

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

build:
	poetry export --without-hashes --format=requirements.txt --output=requirements.txt
	poetry build

publish:
	poetry export --without-hashes --format=requirements.txt --output=requirements.txt
	poetry build
	poetry publish --username __token__ --password $(PYPI_TOKEN)