# stdlib
from pathlib import Path

# third party
from setuptools import find_namespace_packages
from setuptools import setup

package_dir = {}
libs = ["syft"]
packages = []

for p in Path().glob("syft-*/src/*"):
    packages += [pkg for pkg in find_namespace_packages(str(p.parent)) if pkg != "syft"]
    libs.append(p.stem[5:])
    package_dir[f"{p.stem}"] = str(p)

setup(
    name="syft-libs",
    version="0.0.1",
    package_dir=package_dir,
    packages=packages,
    package_data={"": ["package-support.json"]},
    install_requires=libs,
)
