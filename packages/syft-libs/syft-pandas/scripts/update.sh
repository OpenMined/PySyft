#!/bin/bash
jupyter-nbconvert --to python --execute _missing_return/*.ipynb
python scripts/update.py
