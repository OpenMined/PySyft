#!/bin/bash
set -e

./scripts/build_proto.sh

# fix isort and format with black
# isort src/**/*.py tests/**/*.py
black src tests
pre-commit run --all-files
bandit -r src -ll
# pytest -k "not test_all_allowlisted_tensor_methods_work_remotely_on_all_types"
python setup.py test

# run API documentation test notebooks
./scripts/nb_test.sh
pytest examples/api --cov-fail-under 0