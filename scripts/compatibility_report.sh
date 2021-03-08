#!/bin/bash

# check if we are inside a pipenv virtualenv
pipenv --venv
if test $? -eq 0
then
    versions=( "1.6.0" "1.7.0" "1.7.1" )
    for version in "${versions[@]}"
    do
        python .github/adjust_torch_versions.py ./requirements.torch.txt "$version"
        pip install -r requirements.torch.txt
        pytest -m torch --tb=line -n auto
    done

    python tests/syft/lib/allowlist_report.py
else
    echo "Run this inside a pipenv virtualenv"
    exit 1
fi
