#!/bin/bash
jupyter-nbconvert --to python _missing_return/*.ipynb
python scripts/update.py