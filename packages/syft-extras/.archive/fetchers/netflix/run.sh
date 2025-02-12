#!/bin/bash

# Check if chromedriver is in the PATH
if ! command -v chromedriver &> /dev/null
then
    echo "chromedriver is not installed. Installing with brew..."
    brew install chromedriver
else
    echo "chromedriver is already installed."
fi

export CHROMEDRIVER_PATH=$(which chromedriver)
echo $CHROMEDRIVER_PATH

mkdir -p inputs
mkdir -p output

export NETFLIX_EMAIL=$(cat inputs/NETFLIX_EMAIL.txt)
export NETFLIX_PASSWORD=$(cat inputs/NETFLIX_PASSWORD.txt)
export NETFLIX_PROFILE=$(cat inputs/NETFLIX_PROFILE.txt)
export OUTPUT_DIR=$(realpath ./output)

uv pip install -r requirements.txt
uv run main.py
