from setuptools import setup, find_packages

setup(
    name="hagrid",
    description="Happy Automation for Grid",
    version="0.1",
    author="Andrew Trask <andrew@openmined.org>",
    packages=find_packages(),
    install_requires=[
        "click",
    ],
    entry_points={
        "console_scripts": [
            "hagrid = hagrid.cli:cli"
        ]
    },
)