"""This module contains all the external libraries that Syft supports.
    We lazy load the external libraries when they are needed.
"""
# stdlib
import importlib
import os
from typing import Union

# relative
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..util.util import str_to_bool

# Contains all the external libraries that Syft supports.
# Used to check if a library is supported
# if the external library is not installed, we prompt the user
# to install it with the pip package name.

OBLV = str_to_bool(os.getenv("ENABLE_OBLV", "false"))

EXTERNAL_LIBS = {
    "oblv": {
        "pip_package_name": "oblv-ctl",
        "module_name": "oblv_ctl",
    }
}


def package_exists(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def enable_external_lib(lib_name: str) -> Union[SyftSuccess, SyftError]:
    if lib_name in EXTERNAL_LIBS:
        syft_module_name = f"syft.external.{lib_name}"
        pip_package_name = EXTERNAL_LIBS[lib_name]["pip_package_name"]
        if not package_exists(EXTERNAL_LIBS[lib_name]["module_name"]):
            return SyftError(
                message=f"Package: {pip_package_name} for library: {lib_name} not installed.\n"
                + f"Kindly install it with 'pip install {pip_package_name}'"
            )

        importlib.import_module(syft_module_name)
        return SyftSuccess(message=f"Successfully enabled external library: {lib_name}")
    else:
        return SyftError(
            message=f"External library {lib_name} not supported. \n"
            + f"Supported external libraries are: {list(EXTERNAL_LIBS.keys())}"
        )
