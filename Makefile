install-dev:
	# this works around the problem of installing from dev branch with cmdstanpy import
	pip install cmdstanpy
	pip install -r requirements.txt
	# required by mkdocs
	pip install poetry
	pip install -r requirements-test.txt
	pip install -r requirements-docs.txt
	pip install -e .

format:
	black .

lint:
	flake8