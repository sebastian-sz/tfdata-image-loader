format_code:
	black .

install:
	pip install -e .

init_test:
	cd tests; pip install -r requirements.txt

test:
	pytest tests/
