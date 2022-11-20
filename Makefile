install-dev:
	pip install poetry
	poetry install


format:
	black .
	isot .

lint:
	flake8