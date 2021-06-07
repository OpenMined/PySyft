"""
    Setup file for syft.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
# stdlib
from subprocess import check_call

# third party
from setuptools import setup
from setuptools.command.develop import develop


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self) -> None:
        develop.run(self)
        try:
            print("Installing pre-commit and git hooks")
            check_call("pip install pre-commit".split())
            check_call("pre-commit install".split())
        except Exception as e:
            print(f"Failed to install pre-commit. {e}")


if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev", "root": "../.."},
            cmdclass={
                "develop": PostDevelopCommand,
            },
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
