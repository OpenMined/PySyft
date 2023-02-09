#!/bin/bash
if [[ ! -f $(which yarn) ]]; then
    echo 'Please install Yarn: https://yarnpkg.com/getting-started/install'; exit 1
fi
