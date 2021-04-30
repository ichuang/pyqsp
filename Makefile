all:	dist

dist:	
	python setup.py sdist
	python setup.py bdist_wheel --universal

test:
	python3 -m pip install cython
	python3 -m pip install -r tf_requirements.txt
	python3 setup.py install

upload:	
	twine upload dist/*
