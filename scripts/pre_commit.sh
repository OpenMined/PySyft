#!/bin/bash
set -e

# start tests in parallel (with additional parallelism)
pytest -m fast -n auto &

# check bandit in parallel
bandit -r src -ll &

# run API documentation test notebooks in parallel
./scripts/nb_test.sh && pytest examples/api --cov-fail-under 0 &

# fix isort and format with black
./scripts/build_proto.sh && isort . && black src tests && pre-commit run --all-files &
