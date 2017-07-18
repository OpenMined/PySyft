import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "syft",
    version = "0.1.0",
    author = "Amber Trask",
    author_email = "contact@openmined.com",
    description = ("A library for Homomorphically Encrypted Deep Learning Algorithms"),
    license = "Apache-2.0",
    keywords = "deep learning machine artificial intelligence homomorphic encryption",
    packages=['syft', 'test'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Alpha",
    ],
)
