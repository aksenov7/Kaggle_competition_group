venv=python3

install:
	pip3 install virtualenv
	[ -d $(venv) ] || virtualenv -p $(venv) $(venv)

	$(venv)/bin/pip install -U pip
	$(venv)/bin/pip install -U wheel
	$(venv)/bin/pip install -r requirements.txt

test:
	$(venv)/bin/python3 test.py
