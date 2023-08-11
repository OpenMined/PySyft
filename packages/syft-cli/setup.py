# third party
from setuptools import find_packages
from setuptools import setup
from version import __version__

packages = [
    "typer[all]==0.9.0",
    "typing_extensions==4.6.3",
]

dev_packages = [
    "pyinstaller==5.13.0",
]

setup(
    name="Syft CLI",
    description="Command line utility for Syft",
    long_description="",
    long_description_content_type="text/plain",
    version=__version__,
    author="OpenMined <info@openmined.org>",
    packages=find_packages(),
    install_requires=packages,
    extras_require={"dev": dev_packages},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "syft = cli:app",
            "syft-cli = cli:app",
            "syftcli = cli:app",
            "syftctl = cli:app",
        ]
    },
)
