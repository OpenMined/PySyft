# third party
from setuptools import find_packages
from setuptools import setup
from syft_cli.version import __version__

packages = [
    "typer[all]==0.9.0",
    "typing_extensions==4.6.3",
]

build_packages = [
    "pyinstaller==5.13.0",
]

dev_packages = ["pytest"]

setup(
    name="Syft CLI",
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
            "syft = syft_cli.cli:app",
            "syft-cli = syft_cli.cli:app",
            "syftcli = syft_cli.cli:app",
            "syftctl = syft_cli.cli:app",
        ]
    },
)
