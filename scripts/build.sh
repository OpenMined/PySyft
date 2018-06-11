rm -rf build
python setup.py clean install
flake8 . --count --exit-zero --max-complexity=11 --max-line-length=100 --statistics
cd docs
rm -rf ./_modules
rm -rf ./_autosummary
sphinx-apidoc -o ./_modules ../syft
rm -rf _build
make html
cd ../

python setup.py test

# uncomment if you want to open the documentation in a webpage
# open ./docs/_build/html/index.html