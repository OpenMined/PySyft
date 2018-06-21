# Clean up previous build
rm -rf build
python setup.py clean install

# Code Style Checking
flake8 . --count --exit-zero --max-complexity=11 --max-line-length=100 --statistics

# Deletes Documentation
cd docs
rm -rf ./_modules
rm -rf ./_autosummary

# Automatically Generates Documentation
sphinx-apidoc -o ./_modules ../syft
rm -rf _build
make html
cd ../

# Runs Unit Tests
# python setup.py test

# uncomment if you want to open the documentation in a webpage
# open ./docs/_build/html/index.html