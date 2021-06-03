test:
	python -m unittest discover tests/

lint:
	pre-commit run --all-files
