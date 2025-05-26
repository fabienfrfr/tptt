.PHONY: install test clean lint coverage coverage-html build publish docs sys-deps

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
	semantic-release version
	poetry self update
	poetry export --without-hashes --format=requirements.txt --output=requirements.tmp
	sed 's/;.*//' requirements.tmp | grep -v '^[[:space:]]*$$' > requirements.tmp.tmp
	mv requirements.tmp.tmp requirements.tmp
	poetry build

publish:
	poetry export --without-hashes --format=requirements.txt --output=requirements.txt
	sed 's/;.*//' requirements.txt | grep -v '^[[:space:]]*$$' > requirements.txt.tmp
	mv requirements.txt.tmp requirements.txt
	poetry build
	poetry publish --username __token__ --password $(PYPI_TOKEN)

docs:
	cd docs && make html && make clean

sys-deps:
	sudo apt-get update && sudo apt-get install -y \
		build-essential \
		python3-dev \
		libncurses-dev \
		libreadline-dev \
		libsqlite3-dev \
		libbz2-dev \
		libffi-dev \
		libssl-dev \
		libgdbm-dev \
		zlib1g-dev \
		liblzma-dev \
		tk-dev \
		uuid-dev \
		libxml2-dev \
		libxslt1-dev \
		libjpeg-dev \
		libtiff-dev \
		pkg-config \
		xz-utils \
		gdb \
		pkg-config \
		lcov \
		lzma \
		xz-utils