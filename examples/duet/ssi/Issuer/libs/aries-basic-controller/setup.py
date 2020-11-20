"""Module setup."""

import os
import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "aries_basic_controller"
# version_meta = runpy.run_path("./{}/version.py".format(PACKAGE_NAME))
# VERSION = version_meta["__version__"]


with open(os.path.abspath("./README.md"), "r") as fh:
    long_description = fh.read()


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version="0.1.1",
        author="Will Abramson",
        description="A simple python package for controlling an aries agent through the admin-api interface",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/OpenMined/PyDentity/tree/master/libs/aries-basic-controller",
        packages=find_packages(),
        include_package_data=True,
        package_data={"aries_basic_controller": ["requirements.txt"]},
        install_requires=parse_requirements("requirements.txt"),
        python_requires=">=3.6.3",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
    )