all:	dist

dist:	
	python setup.py sdist
	python setup.py bdist_wheel --universal

upload:	
	twine upload dist/*
