# third party
from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.0"

if __name__ == "__main__":
    setup(
        name="simtester",
        version=__version__,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        entry_points={
            "console_scripts": [
                "simtester=simtester.__main__:main",  # Exposes the command 'simtester'
            ],
        },
    )
