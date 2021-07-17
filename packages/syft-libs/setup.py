# stdlib
from pathlib import Path

# third party
import setuptools
from setuptools import find_namespace_packages

package_dir = {}
packages = []
libs = ["syft"]

for p in Path().glob("*/src/syft/*"):
    packages += [
        pkg for pkg in find_namespace_packages(str(p.parent.parent)) if pkg != "syft"
    ]
    libs.append(p.stem)
    package_dir[f"syft.{p.stem}"] = str(p)

print(package_dir, packages, libs)

setuptools.setup(
    name="syft.libs",
    version="0.0.1",
    package_dir=package_dir,
    packages=packages,
    package_data={"": ["package-support.json"]},
    install_requires=libs,
)
