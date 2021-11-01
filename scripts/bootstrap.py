#!/usr/bin/env python3
# stdlib
import os
import platform

p = platform.system().lower().replace("darwin", "macos")

if p == "macos":
    os.system("type pyenv &> /dev/null || brew install pyenv")
else:
    os.system("type pyenv &> /dev/null || curl https://pyenv.run | bash")
