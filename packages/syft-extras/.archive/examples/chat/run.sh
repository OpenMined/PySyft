#!/bin/sh
uv venv
uv pip install -r requirements.txt
uv run main.py
