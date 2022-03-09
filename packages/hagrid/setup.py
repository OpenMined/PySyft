# third party
from hagrid import __version__
from setuptools import find_packages
from setuptools import setup

DATA_FILES = {"img": ["hagrid/img/*.png"]}

setup(
    name="hagrid",
    description="Happy Automation for Grid",
    version=__version__,
    author="Andrew Trask <andrew@openmined.org>",
    packages=find_packages(),
    package_data=DATA_FILES,
    install_requires=[
        "ascii_magic",
        "click",
        "gitpython",
        "names",
        "requests",
        "rich",
        "setuptools",
    ],
    include_package_data=True,
    entry_points={"console_scripts": ["hagrid = hagrid.cli:cli"]},
)
