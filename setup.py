import os

from setuptools import find_packages
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


REQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pip-dep")
core_reqs = read(os.path.join(REQ_DIR, "requirements.txt")).split("\n")
udacity_extras = read(os.path.join(REQ_DIR, "requirements_udacity.txt")).split("\n")
tensorflow_extras = read(os.path.join(REQ_DIR, "requirements_tensorflow.txt")).split("\n")
sandbox_extras = ["scikit-learn>=0.21.0"]


setup(
    name="syft",
    version="0.2.0a2",
    author="Andrew Trask",
    author_email="contact@openmined.org",
    description=("A Library for Private, Secure Deep Learning"),
    license="Apache-2.0",
    keywords="deep learning artificial intelligence privacy secure multi-party computation federated learning differential privacy",
    packages=find_packages(exclude=["docs", "examples", "dist"]),
    include_package_data=True,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/OpenMined/PySyft",
    install_requires=core_reqs,
    extras_require={
        "udacity": udacity_extras,
        "sandbox": sandbox_extras,
        "tensorflow": tensorflow_extras,
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-flake8"],
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
)
