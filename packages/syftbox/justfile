# Guidelines for new commands
# - Start with a verb
# - Keep it short (max. 3 words in a command)
# - Group commands by context. Include group name in the command name.
# - Mark things private that are util functions with [private] or _var
# - Don't over-engineer, keep it simple.
# - Don't break existing commands
# - Run just --fmt --unstable after adding new commands

set dotenv-load := true

# ---------------------------------------------------------------------------------------------------------------------
# Private vars

_red := '\033[1;31m'
_cyan := '\033[1;36m'
_green := '\033[1;32m'
_yellow := '\033[1;33m'
_nc := '\033[0m'

# ---------------------------------------------------------------------------------------------------------------------
# Aliases

alias rs := run-server
alias rsu := run-server-uvicorn
alias rc := run-client
alias rj := run-jupyter
alias b := build

# ---------------------------------------------------------------------------------------------------------------------

@default:
    just --list

# ---------------------------------------------------------------------------------------------------------------------

# Run a local syftbox server on port 5001
[group('server')]
run-server port="5001" gunicorn_args="":
    #!/bin/bash
    set -eou pipefail

    export SYFTBOX_DATA_FOLDER=${SYFTBOX_DATA_FOLDER:-.server/data}
    uv run syftbox server migrate
    uv run gunicorn syftbox.server.server:app -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:{{ port }} --reload {{ gunicorn_args }}

migrate:
    #!/bin/bash
    set -eou pipefail

    uv run syftbox server migrate

[group('server')]
run-server-uvicorn port="5001" uvicorn_args="":
    #!/bin/bash
    set -eou pipefail

    export SYFTBOX_DATA_FOLDER=${SYFTBOX_DATA_FOLDER:-.server/data}
    uv run syftbox server migrate
    uv run uvicorn syftbox.server.server:app --host 127.0.0.1 --port {{ port }} --reload --reload-dir ./syftbox {{ uvicorn_args }}

# ---------------------------------------------------------------------------------------------------------------------

# Run a local syftbox client on any available port between 8080-9000
[group('client')]
run-client name port="auto" server="http://localhost:5001":
    #!/bin/bash
    set -eou pipefail

    # generate a local email from name, but if it looks like an email, then use it as is
    EMAIL="{{ name }}@openmined.org"
    if [[ "{{ name }}" == *@*.* ]]; then EMAIL="{{ name }}"; fi

    # if port is auto, then generate a random port between 8000-8090, else use the provided port
    PORT="{{ port }}"
    if [[ "$PORT" == "auto" ]]; then PORT="0"; fi

    # Working directory for client is .clients/<email>
    DATA_DIR=.clients/$EMAIL
    mkdir -p $DATA_DIR

    echo -e "Email      : {{ _green }}$EMAIL{{ _nc }}"
    echo -e "Client     : {{ _cyan }}http://localhost:$PORT{{ _nc }}"
    echo -e "Server     : {{ _cyan }}{{ server }}{{ _nc }}"
    echo -e "Data Dir   : $DATA_DIR"

    uv run syftbox client --config=$DATA_DIR/config.json --data-dir=$DATA_DIR --email=$EMAIL --port=$PORT --server={{ server }} --no-open-dir

# ---------------------------------------------------------------------------------------------------------------------

[group('client')]
run-live-client server="https://syftbox.openmined.org/":
    uv run syftbox client --server={{ server }}

# ---------------------------------------------------------------------------------------------------------------------

# Run a local syftbox app command
[group('app')]
run-app name command subcommand="":
    #!/bin/bash
    set -eou pipefail

    # generate a local email from name, but if it looks like an email, then use it as is
    EMAIL="{{ name }}@openmined.org"
    if [[ "{{ name }}" == *@*.* ]]; then EMAIL="{{ name }}"; fi

    # Working directory for client is .clients/<email>
    DATA_DIR=$(pwd)/.clients/$EMAIL
    mkdir -p $DATA_DIR
    echo -e "Data Dir   : $DATA_DIR"

    uv run syftbox/main.py app {{ command }} {{ subcommand }} --config=$DATA_DIR/config.json

# ---------------------------------------------------------------------------------------------------------------------

# Build syftbox wheel
[group('build')]
build:
    rm -rf dist
    uv build


# Build syftbox wheel
[group('install')]
install:
    rm -rf dist
    uv build
    uv tool install $(ls ./dist/*.whl) --reinstall

# Bump version, commit and tag
[group('build')]
bump-version level="patch" breaking_changes="false":
    #!/bin/bash
    # We need to uv.lock before we can commit the whole thing in the repo.
    # DO not bump the version on the uv.lock file, else other packages with same version might get updated

    set -eou pipefail

    # sync dev dependencies for bump2version
    uv sync --frozen

    # get the current and new version
    BUMPVERS_CHANGES=$(uv run bump2version --dry-run --allow-dirty --list {{ level }})
    CURRENT_VERSION=$(echo "$BUMPVERS_CHANGES" | grep current_version | cut -d'=' -f2)
    NEW_VERSION=$(echo "$BUMPVERS_CHANGES" | grep new_version | cut -d'=' -f2)
    echo "Bumping version from $CURRENT_VERSION to $NEW_VERSION"

    # first bump version
    uv run bump2version {{ level }}

    # upgrade version compatibility matrix
    cd scripts
    BREAKING_CHANGES=""
    if [[ '{{ breaking_changes }}' == true ]]; then BREAKING_CHANGES="--breaking_changes"; fi
    uv run upgrade_version_matrix.py {{ level }} $BREAKING_CHANGES
    cd ..
    # update uv.lock file to reflect new package version
    uv lock

    # commit the changes
    git commit -am "Bump version $CURRENT_VERSION -> $NEW_VERSION"
    git tag -a $NEW_VERSION -m "Release $NEW_VERSION"

# ---------------------------------------------------------------------------------------------------------------------

[group('test')]
test-e2e-old test_name:
    @echo "Using SyftBox from {{ _green }}'$(which syftbox)'{{ _nc }}"
    chmod +x ./tests/e2e/{{ test_name }}/run.bash
    bash ./tests/e2e.old/{{ test_name }}/run.bash

[group('test')]
test-e2e test_name:
    #!/bin/sh
    uv sync --frozen
    . .venv/bin/activate
    echo "Using SyftBox from {{ _green }}'$(which syftbox)'{{ _nc }}"
    pytest -sq --color=yes ./tests/e2e/test_{{ test_name }}.py

# ---------------------------------------------------------------------------------------------------------------------

# Build & Deploy syftbox to a remote server using SSH
[group('deploy')]
upload-dev keyfile remote="user@0.0.0.0": build
    #!/bin/bash
    set -eou pipefail

    # there will be only one wheel file in the dist directory, but you never know...
    LOCAL_WHEEL=$(ls dist/*.whl | grep syftbox | head -n 1)

    # Remote paths to copy the wheel to
    REMOTE_DIR="~"
    REMOTE_WHEEL="$REMOTE_DIR/$(basename $LOCAL_WHEEL)"

    echo -e "Deploying {{ _cyan }}$LOCAL_WHEEL{{ _nc }} to {{ _green }}{{ remote }}:$REMOTE_WHEEL{{ _nc }}"

    # change permissions to comply with ssh/scp
    chmod 600 {{ keyfile }}

    # Use scp to transfer the file to the remote server
    scp -i {{ keyfile }} "$LOCAL_WHEEL" "{{ remote }}:$REMOTE_DIR"

    # install pip package
    ssh -i {{ keyfile }} {{ remote }} "uv venv && uv pip install $REMOTE_WHEEL"

    # restart service
    # NOTE - syftbox service is created manually on the remote server
    ssh -i {{ keyfile }} {{ remote }} "sudo systemctl daemon-reload && sudo systemctl restart syftbox"
    echo -e "{{ _green }}Deployed SyftBox local wheel to {{ remote }}{{ _nc }}"

# Deploy syftbox from pypi to a remote server using SSH
[group('deploy')]
upload-pip version keyfile remote="user@0.0.0.0":
    #!/bin/bash
    set -eou pipefail

    # change permissions to comply with ssh/scp
    chmod 600 {{ keyfile }}

    echo -e "Deploying syftbox version {{ version }} to {{ remote }}..."

    # install pip package
    ssh -i {{ keyfile }} {{ remote }} "uv venv && uv pip install syftbox=={{ version }}"

    # restart service
    ssh -i {{ keyfile }} {{ remote }} "sudo systemctl daemon-reload && sudo systemctl restart syftbox"

    echo -e "{{ _green }}Deployed SyftBox {{ version }} to {{ remote }}{{ _nc }}"

# ---------------------------------------------------------------------------------------------------------------------

[group('utils')]
ssh keyfile remote="user@0.0.0.0":
    ssh -i {{ keyfile }} {{ remote }}

# remove all local files & directories
[group('utils')]
reset:
    rm -rf ./.clients ./.server ./dist ./.e2e

[group('utils')]
run-jupyter jupyter_args="":
    uv run --frozen --with "jupyterlab" \
        jupyter lab {{ jupyter_args }}

auth email server="http://127.0.0.1:5001":
    # get access token from dev server

    EMAIL={{ email }} && \
    EMAIL_TOKEN=$( \
        curl -s -X 'POST' '{{server}}/auth/request_email_token' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d "{\"email\": \"${EMAIL}\"}" \
        | jq -r '.email_token' \
    ) && \
    curl -s -X 'POST' "{{server}}/auth/validate_email_token?email=${EMAIL}" \
        -H 'accept: application/json' \
        -H "Authorization: Bearer ${EMAIL_TOKEN}" \
        -d '' \
    | jq -r '.access_token'