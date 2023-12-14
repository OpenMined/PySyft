# stdlib
import platform

# third party
from setuptools import find_packages
from setuptools import setup

__version__ = "0.3.97"

DATA_FILES = {"img": ["hagrid/img/*.png"], "hagrid": ["*.yml"]}

packages = [
    "ascii_magic",
    "click>=8.1.7",
    "cryptography>=41.0.4",
    "gitpython",
    "jinja2",
    "names",
    "packaging>=23.0",
    "paramiko",
    "pyOpenSSL>=23.2.0",
    "requests",
    "rich",
    "setuptools",
    "virtualenv-api",
    "virtualenv",
    "PyYAML",
    "tqdm",
    "gevent>=22.10.2,<=23.9.1",
]

if platform.system().lower() != "windows":
    packages.extend(["ansible", "ansible-core"])

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
