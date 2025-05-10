.PHONY: install test clean

install:
	pip install -r requirements.txt

test:
	@export $(shell cat .env | xargs) && pytest

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +