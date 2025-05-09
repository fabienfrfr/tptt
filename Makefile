.PHONY: install test clean

install:
	pip install -r requirements.txt

test:
	pytest

clean:
	rm -rf build dist *.egg-info .pytest_cache
