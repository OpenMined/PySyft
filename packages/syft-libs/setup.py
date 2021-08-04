# stdlib
import os
import platform

# third party
from setuptools import find_packages
from setuptools import setup

packages = ["syft"]

here = os.path.abspath(os.path.dirname(__file__))
local = os.path.exists(f"{here}/pyscaffoldext-syft-support")

def check_petlib_windows(package:str)->bool:
    p = platform.system().lower()
    if p == "windows" and package.endswith("petlib"):
        return True 
    return False

if local:
    for package in os.listdir(here):
        if package.startswith("syft-") and not check_petlib_windows(package):
            local_package = (
                f"{package} @ file://localhost{here}/{package}#egg={package}"
            )
            packages.append(local_package)
else:
    packages += ["syft-xgboost", "syft-statsmodels", "syft-opacus", "syft-petlib"]
    if platform.system().lower() == "windows":
        packages.remove("syft-petlib")

packages = sorted(list(packages))

setup(
    name="syft-libs",
    version="0.0.1",
    packages=find_packages(),
    install_requires=packages,
)
