install-dev:
	pip install -e .

install-poetry:
	pip install poetry
	poetry install


format:
	black .
	isot .

lint:
	flake8