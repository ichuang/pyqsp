PYTHON ?= python

.PHONY: all core dist build test lint upload

all core dist: build

build:
	$(PYTHON) -m build --no-isolation

test:
	MPLBACKEND=Agg $(PYTHON) -m pytest pyqsp/test

lint:
	$(PYTHON) -m flake8 . --select=E9,F63,F7,F82 --show-source

upload:
	$(PYTHON) -m twine upload dist/*
