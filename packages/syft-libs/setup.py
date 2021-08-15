# stdlib
import os

# third party
from setuptools import find_packages, setup

packages = ["syft"]

here = os.path.abspath(os.path.dirname(__file__))
local = os.path.exists(f"{here}/pyscaffoldext-syft-support")

if local:
    for package in os.listdir(here):
        if package.startswith("syft-"):
            local_package = (
                f"{package} @ file://localhost{here}/{package}#egg={package}"
            )
            packages.append(local_package)
else:
    packages += ["syft-xgboost", "syft-statsmodels"]

packages = sorted(list(packages))

setup(
    name="syft-libs",
    version="0.0.1",
    packages=find_packages(),
    install_requires=packages,
)
