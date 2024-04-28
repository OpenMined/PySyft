#!/usr/bin/env python3

import setuptools
import os
import subprocess

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    setuptools.setup()
    try:
        import google.colab
        install('numpy==1.25.2')  # version for Google Colab
    except ImportError:
        pass  # version specified in setup.cfg will be used for other environments
