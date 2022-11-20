install-dev:
	pip install pystan==2.19.1.1
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

format:
	black .

lint:
	flake8