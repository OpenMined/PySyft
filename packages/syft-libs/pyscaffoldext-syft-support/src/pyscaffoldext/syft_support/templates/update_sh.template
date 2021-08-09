#!/bin/bash
jupyter-nbconvert --to python --execute _missing_return/*.ipynb 
python scripts/update.py
find _missing_return -name "*.py" -and ! -name "__init__.py" -delete