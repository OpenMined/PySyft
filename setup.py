# -*- coding: utf-8 -*-
"""
    Setup file for syft.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

# stdlib
from subprocess import check_call
import sys

# third party
from pkg_resources import VersionConflict
from pkg_resources import require
from setuptools import setup
from setuptools.command.develop import develop


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self) -> None:
        develop.run(self)
        try:
            check_call("pip install pre-commit".split())
            check_call("pre-commit install".split())
        except Exception as e:
            print(f"Failed to install pre-commit. {e}")


try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        url="https://github.com/OpenMined/PySyft",
        cmdclass={
            "develop": PostDevelopCommand,
        },
    )
