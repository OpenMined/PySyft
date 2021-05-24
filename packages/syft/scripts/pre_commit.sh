#!/bin/bash
set -e

# collect the process ids
pids=()

# start tests in parallel (with additional parallelism)
pytest -m fast -n auto & pids+=($!)

# check bandit in parallel
bandit -r src & pids+=($!)

# run API documentation test notebooks in parallel
./scripts/nb_test.sh && pytest examples/api --cov-fail-under 0 & pids+=($!)

# fix isort and format with black
./scripts/build_proto.sh && isort . && black src tests && pre-commit run --all-files & pids+=($!)

# check for error return codes
error=0
for pid in ${pids[*]}; do
    if ! wait $pid; then
        error=1
    fi
done

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if [ "$error" -eq "0" ]; then
    printf "\n> Pre Commit ${GREEN}PASSED${NC}\n"
else
    printf "\n> Pre Commit ${RED}FAILED${NC}\n"
fi;

exit $error
