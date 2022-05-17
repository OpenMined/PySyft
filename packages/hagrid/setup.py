# stdlib
import platform

# third party
from hagrid import __version__
from setuptools import find_packages
from setuptools import setup

DATA_FILES = {
    "img": ["hagrid/img/*.png"],
}

packages = [
    "ascii_magic",
    "click",
    "gitpython",
    "names",
    "requests",
    "rich",
    "setuptools",
    "cryptography>=37.0.2",
    "pyOpenSSL>=22.0.0",
]

if platform.system().lower() != "windows":
    packages.extend(["ansible", "ansible-core"])

# Pillow binary wheels for Apple Silicon on Python 3.8 don't seem to work well
# try using Python 3.9+ for HAGrid on Apple Silicon
setup(
    name="hagrid",
    description="Happy Automation for Grid",
    long_description="HAGrid is the swiss army knife of OpenMined's PySyft and PyGrid.",
    long_description_content_type="text/plain",
    version=__version__,
    author="Andrew Trask <andrew@openmined.org>",
    packages=find_packages(),
    package_data=DATA_FILES,
    install_requires=packages,
    include_package_data=True,
    entry_points={"console_scripts": ["hagrid = hagrid.cli:cli"]},
)
