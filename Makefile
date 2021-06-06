SRC = $(wildcard ./*.ipynb)

all: rad_transformer docs

rad_transformer: $(SRC)
	nbdev_build_lib
	touch rad_transformer

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs
    
test_all:
	nbdev_test_nbs --pause 0.5 --flags 'cuda slow'
    
release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist