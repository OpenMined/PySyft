import os
import sys
import platform
import subprocess

from setuptools import setup

# We need to add our rest api as a path since it is a separate application
# deployed on Heroku:
path = os.path.dirname(os.path.abspath(__file__)) + "/app/pg_rest_api"
sys.path.insert(0, path)
platform = platform.system()

setup(setup_requires=["pbr", "pytest-runner"], pbr=True)
