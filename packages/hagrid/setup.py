# stdlib
import platform

# third party
from setuptools import find_packages
from setuptools import setup

__version__ = "0.2.114"

DATA_FILES = {"img": ["hagrid/img/*.png"], "hagrid": ["*.yml"]}

packages = [
    "ascii_magic==1.6",
    "click==8.1.3",
    "cryptography==38.0.4",
    "gitpython==3.1.29",
    "jinja2==3.1.2",
    "names==0.3.0",
    "packaging==22.0",
    "paramiko==2.12.0",
    "pyOpenSSL==22.1.0",
    "requests==2.28.1",
    "rich==11.1.0",
    "setuptools==65.6.3",
    "virtualenv-api==2.1.18",
    "virtualenv==20.17.1",
    "PyYAML==6.0",
    "tqdm==4.64.1",
]

if platform.system().lower() != "windows":
    packages.extend(["ansible==7.1.0", "ansible-core==2.14.1"])

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
