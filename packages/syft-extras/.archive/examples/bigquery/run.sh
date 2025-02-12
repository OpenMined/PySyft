#!/bin/sh
uv venv
uv pip install -r requirements.txt
uv run main.py
# bore local 9081 --to bore.pub > bore_output.log 2>&1 &
