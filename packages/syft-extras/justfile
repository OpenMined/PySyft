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

alias ba := build-all
alias rj := run-jupyter

# ---------------------------------------------------------------------------------------------------------------------

@default:
    just --list

[group('build')]
build-all:
    uv build --all-packages

[group('utils')]
run-jupyter jupyter_args="":
    uv sync

    uv run --frozen --with "jupyterlab" \
        jupyter lab {{ jupyter_args }}

start-proxy *args:
    uv sync
    rm -rf certs
    sudo uv run syft_proxy bootstrap
    uv run syft_proxy start {{ args }}

run-pong:
    uv sync
    uv run examples/pingpong/pong_server.py

run-ping:
    uv sync
    uv run examples/pingpong/ping_request.py
