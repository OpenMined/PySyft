# third party
from setuptools import find_packages
from setuptools import setup

DATA_FILES = {
    "img": ["hagrid/img/*.png"],
}

setup(
    name="hagrid",
    description="Happy Automation for Grid",
    version="0.1.3",
    author="Andrew Trask <andrew@openmined.org>",
    packages=find_packages(),
    package_data=DATA_FILES,
    install_requires=[
        "ansible-core",
        "ansible",
        "ascii_magic",
        "azure-cli",
        "click",
        "gitpython",
        "names",
        "paramiko",
        "requests",
        "rich",
        "setuptools",
    ],
    include_package_data=True,
    entry_points={"console_scripts": ["hagrid = hagrid.cli:cli"]},
)
