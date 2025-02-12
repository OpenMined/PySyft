#!/bin/sh

# Detect if macOS or something else (e.g., Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    uv run --python 3.11 -- uv pip install ./Pyfhel-3.4.2-cp311-cp311-macosx_13_0_arm64.whl
else
    uv run --python 3.11 -- uv pip install pyfhel==3.4.2
fi

uv run --python 3.11 -- uv run main.py "$@"
