# stdlib
import imp
import os

# third party
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.md file
with open(os.path.join(here, "DEPLOYMENT.md"), encoding="utf-8") as f:
    long_description = f.read()

VERSION = imp.load_source("VERSION", "../../VERSION")

setup(
    name="pygrid-cli",
    version=VERSION.__version__,
    description="OpenMined PyGrid CLI for Infrastructure and cloud deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpenMined/PyGrid",
    packages=[
        "apps.domain.src.main.core.infrastructure",
        "apps.domain.src.main.core.infrastructure.providers",
        "apps.domain.src.main.core.infrastructure.providers.aws",
        "apps.domain.src.main.core.infrastructure.providers.azure",
        "apps.domain.src.main.core.infrastructure.providers.gcp",
    ],
    install_requires=["click", "PyInquirer", "terrascript", "boto3", "flask"],
    entry_points="""
        [console_scripts]
        pygrid=apps.domain.src.main.core.infrastructure.cli:cli
    """,
)
