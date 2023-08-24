#! /bin/bash

# pyinstaller code here
rm -rf build/ dist/
pyinstaller --onefile src/cli.py --name=syftcli

