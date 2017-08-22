import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements = read('requirements.txt').split()

setup(
    name="syft",
    version="0.1.0",
    author="Amber Trask",
    author_email="contact@openmined.org",
    description=("A library for Encrypted Deep Learning Algorithms"),
    license="Apache-2.0",
    keywords="deep learning artificial intelligence homomorphic encryption",
    packages=find_packages(exclude=['notebooks', 'test*', 'dist']),
    include_package_data=True,
    long_description=read('README.md'),
    url='github.com/OpenMined/Syft',
    classifiers=[
        "Development Status :: 1 - Alpha",
    ],
    scripts=['bin/syft_cmd'],
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-flake8']
)
