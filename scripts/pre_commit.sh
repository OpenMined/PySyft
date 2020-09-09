#!/bin/bash
set -e

./scripts/build_proto.sh

# fix isort and format with black
isort .
black src tests
pre-commit run --all-files
bandit -r src -ll
pytest -k "not test_all_allowlist"
# python setup.py test

# run API documentation test notebooks
./scripts/nb_test.sh
pytest examples/api --cov-fail-under 0