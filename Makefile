format_code:
	black .

install:
	pip install -e .

test:
	python -m unittest discover tests/

lint:
	flake8 --max-line-length=88 .