#!/bin/bash
if [[ ! -f $(which pnpm) ]]; then
    echo 'Please install pnpm: https://pnpm.io/installation'; exit 1
fi
