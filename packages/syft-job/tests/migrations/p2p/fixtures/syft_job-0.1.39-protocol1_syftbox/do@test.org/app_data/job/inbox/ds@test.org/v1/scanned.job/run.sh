#!/bin/bash
set -euo pipefail
export UV_SYSTEM_PYTHON=false
cd code
uv venv --python 3.12
source .venv/bin/activate
uv pip install "syft-client" "pandas"
export PYTHONPATH=.:${PYTHONPATH:-}
python main.py
