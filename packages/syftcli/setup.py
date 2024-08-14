# third party
from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.11"

packages = [
    "requests==2.32.3",
    "pyyaml==6.0.1",
    "packaging==24.1",
    "typer[all]==0.12.3",
    "typing_extensions==4.12.2",
]

build_packages = [
    "pyinstaller==6.10.0",
]

dev_packages = ["pytest"] + build_packages

setup(
    name="syftcli",
    description="Command line utility for Syft",
    long_description="",
    long_description_content_type="text/plain",
    version=__version__,
    author="OpenMined <info@openmined.org>",
    packages=find_packages(),
    install_requires=packages,
    extras_require={"dev": dev_packages, "build": build_packages},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "syft = syftcli.cli:app",
            "syft-cli = syftcli.cli:app",
            "syftcli = syftcli.cli:app",
            "syftctl = syftcli.cli:app",
        ]
    },
)
