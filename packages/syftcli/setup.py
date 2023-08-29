# third party
from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.4"

packages = [
    "typer[all]==0.9.0",
    "typing_extensions==4.6.3",
]

build_packages = [
    "pyinstaller==5.13.0",
]

dev_packages = ["pytest"] + build_packages

setup(
    name="syftcli",
    description="Command line utility for Syft",
    long_description="",
    long_description_content_type="text/plain",
    version=__version__,
    author="OpenMined <info@openmined.org>",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
