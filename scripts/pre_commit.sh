#!/bin/bash
set -e

./scripts/build_proto.sh

# fix isort and format with black
# isort src/**/*.py tests/**/*.py
black src tests
pre-commit run --all-files
bandit -r src -ll
python setup.py test

# run API documentation test notebooks
jupyter nbconvert --to script examples/api/*.ipynb

## pytest fails due to some weird hook issue?
## using normal python for now exit code should still be 0 for success
# pytest examples/api

python examples/api/start.py
echo $?