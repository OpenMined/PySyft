import os

from setuptools import find_packages
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements = read("requirements.txt").split()


setup(
    name="syft",
    version="0.1.0",
    author="Andrew Trask",
    author_email="contact@openmined.org",
    description=("A Library for Private, Secure Deep Learning"),
    license="Apache-2.0",
    keywords="deep learning artificial intelligence homomorphic encryption",
    packages=find_packages(exclude=["docs", "examples", "dist"]),
    include_package_data=True,
    long_description=read("README.md"),
    url="github.com/OpenMined/PySyft",
    classifiers=["Development Status :: 1 - Alpha"],
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-flake8"],
)
