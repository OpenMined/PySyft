#!/bin/sh
uv run main.py $( [ "$1" = "--private" ] && echo '--private' )
