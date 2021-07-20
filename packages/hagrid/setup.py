# third party
from setuptools import find_packages
from setuptools import setup

DATA_FILES = {
    "img": ["hagrid/img/*.png"],
}

setup(
    name="hagrid",
    description="Happy Automation for Grid",
    version="0.1.1",
    author="Andrew Trask <andrew@openmined.org>",
    packages=find_packages(),
    package_data=DATA_FILES,
    install_requires=[
        "click",
        "names",
        "requests",
        "setuptools",
        "ansible",
        "ansible-core",
        "paramiko",
        "rich",
        "ascii_magic",
        "gitpython",
    ],
    include_package_data=True,
    entry_points={"console_scripts": ["hagrid = hagrid.cli:cli"]},
)
