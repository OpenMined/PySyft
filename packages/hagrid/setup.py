# third party
from hagrid import __version__
from setuptools import find_packages
from setuptools import setup

DATA_FILES = {
    "img": ["hagrid/img/*.png"],
}

# Pillow binary wheels for Apple Silicon on Python 3.8 don't seem to work well
# try using Python 3.9+ for HAGrid on Apple Silicon
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
