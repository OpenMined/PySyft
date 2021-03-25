# import pathlib
from setuptools import setup

setup(
    name="OpenMined PyGrid CLI",
    version="0.0.1",
    packages=["apps.domain.src.main.core.infrastructure"],
    install_requires=["click", "PyInquirer", "terrascript", "boto3"],
    entry_points="""
        [console_scripts]
        pygrid=apps.domain.src.main.core.infrastructure.cli:cli
    """,
)
