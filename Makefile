all:
	python -m pip install cython
	python -m pip install -r base_requirements.txt
	python setup.py sdist
	python setup.py bdist_wheel --universal

core:
	dist

dist:	
	python setup.py sdist
	python setup.py bdist_wheel --universal

test:
	python -m pip install cython
	python -m pip install -r base_requirements.txt
	python setup.py install

upload:	
	twine upload dist/*
