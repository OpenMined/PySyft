"""Module setup."""

import os
import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "attachment_controller"
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
        version="0.0.1",
        author="Will Abramson",
        description="A python package for an OpenMined specific Aries Controller",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # url="https://github.com/wip-abramson/aries-basic-controller-python",
        packages=find_packages(),
        include_package_data=True,
        package_data={"attachment_controller": ["requirements.txt"]},
        install_requires=parse_requirements("requirements.txt"),
        python_requires=">=3.6.3",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
    )