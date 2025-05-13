.PHONY: install test clean lint coverage coverage-html

install:
	pip install -r requirements.txt

lint:
	isort liza/ tests/
	python -m pylint liza/ tests/

test:
	@export $(shell cat .env | xargs) && pytest --tb=short -v

coverage:
	pytest --cov=liza --cov-report=term-missing

coverage-html:
	pytest --cov=liza --cov-report=html

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +