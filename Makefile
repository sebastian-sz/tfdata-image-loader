format_code:
	black .

install:
	pip install -e .

test:
	python -m unittest discover tests/

lint:
	pre-commit run --all-files
