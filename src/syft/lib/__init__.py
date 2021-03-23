# stdlib
import importlib
import sys
from typing import Any
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Optional
import warnings

# third party
from packaging import version

# syft relative
from ..ast.globals import Globals
from ..lib.petlib import create_petlib_ast
from ..lib.plan import create_plan_ast
from ..lib.python import create_python_ast
from ..lib.torch import create_torch_ast
from ..lib.torchvision import create_torchvision_ast
from ..lib.zksk import create_zksk_ast
from ..logger import critical
from ..logger import traceback_and_raise
from ..logger import warning
from .misc import create_union_ast


class VendorLibraryImportException(Exception):
    pass


def vendor_requirements_available(vendor_requirements: TypeDict[str, TypeAny]) -> bool:
    """
    Check whether torch or python version is supported

    Args:
        vendor_requirements: dictionary containing version of python or torch to be supported

    Returns:
        True if system supports all vendor requirements

    """
    # see if python version is supported
    if "python" in vendor_requirements:
        python_reqs = vendor_requirements["python"]

        PYTHON_VERSION = sys.version_info
        min_version = python_reqs.get("min_version", None)
        if min_version is not None:
            if PYTHON_VERSION < min_version:
                traceback_and_raise(
                    VendorLibraryImportException(
                        f"Unable to load {vendor_requirements['lib']}."
                        + f"Python: {PYTHON_VERSION} < {min_version}"
                    )
                )
        max_version = python_reqs.get("max_version", None)
        if max_version is not None:
            if PYTHON_VERSION > max_version:
                traceback_and_raise(
                    VendorLibraryImportException(
                        f"Unable to load {vendor_requirements['lib']}."
                        + f"Python: {PYTHON_VERSION} > {max_version}"
                    )
                )

    # see if torch version is supported
    if "torch" in vendor_requirements:
        torch_reqs = vendor_requirements["torch"]
        # third party
        import torch

        TORCH_VERSION = version.parse(torch.__version__.split("+")[0])
        min_version = torch_reqs.get("min_version", None)
        if min_version is not None:
            if TORCH_VERSION < version.parse(min_version):
                traceback_and_raise(
                    VendorLibraryImportException(
                        f"Unable to load {vendor_requirements['lib']}."
                        + f"Torch: {TORCH_VERSION} < {min_version}"
                    )
                )

        max_version = torch_reqs.get("max_version", None)
        if max_version is not None:
            if TORCH_VERSION > version.parse(max_version):
                traceback_and_raise(
                    VendorLibraryImportException(
                        f"Unable to load {vendor_requirements['lib']}."
                        + f"Torch: {TORCH_VERSION} > {max_version}"
                    )
                )

    return True


def _load_lib(lib: str, options: TypeDict[str, TypeAny] = {}) -> None:
    """
    Load and Update Node with given library module

    Args:
        lib: name of library to load and update Node with
        options: external requirements for loading library successfully
    """
    _ = importlib.import_module(lib)
    vendor_ast = importlib.import_module(f"syft.lib.{lib}")
    PACKAGE_SUPPORT = getattr(vendor_ast, "PACKAGE_SUPPORT", None)
    PACKAGE_SUPPORT.update(options)
    if PACKAGE_SUPPORT is not None and vendor_requirements_available(
        vendor_requirements=PACKAGE_SUPPORT
    ):
        update_ast = getattr(vendor_ast, "update_ast", None)
        if update_ast is not None:
            global lib_ast
            update_ast(ast_or_client=lib_ast)

            for _, client in lib_ast.registered_clients.items():
                update_ast(ast_or_client=client)

            # cache the constructor for future created clients
            lib_ast.loaded_lib_constructors[lib] = update_ast


def load(lib: str, options: TypeDict[str, TypeAny] = {}) -> None:
    """
    Load and Update Node with given library module

    Args:
        lib: name of library to load and update Node with
        options: external requirements for loading library successfully
    """
    try:
        _load_lib(lib=lib, options=options)
    except VendorLibraryImportException as e:
        critical(e)
    except Exception as e:
        critical(f"Unable to load package support for: {lib}. {e}")


def load_lib(lib: str, options: TypeDict[str, TypeAny] = {}) -> None:
    """
    Load and Update Node with given library module
    load_lib() is deprecated please use load() in the future

    Args:
        lib: name of library to load and update Node with
        options: external requirements for loading library successfully

    """
    msg = "load_lib() is deprecated please use load() in the future"
    warning(msg, print=True)
    warnings.warn(msg, DeprecationWarning)
    load(lib=lib, options=options)


# now we need to load the relevant frameworks onto the node
def create_lib_ast(client: Optional[Any] = None) -> Globals:
    """
    Create AST and load the relevant frameworks onto the node

    Args:
        client: VM client onto whom the frameworks need to be loaded

    Returns:
        AST for client of type Globals

    """
    python_ast = create_python_ast(client=client)
    torch_ast = create_torch_ast(client=client)
    torchvision_ast = create_torchvision_ast(client=client)
    zksk_ast = create_zksk_ast(client=client)
    petlib_ast = create_petlib_ast(client=client)
    # numpy_ast = create_numpy_ast()
    plan_ast = create_plan_ast(client=client)

    lib_ast = Globals(client=client)
    lib_ast.add_attr(attr_name="syft", attr=python_ast.attrs["syft"])
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])
    lib_ast.add_attr(attr_name="torchvision", attr=torchvision_ast.attrs["torchvision"])
    lib_ast.add_attr(attr_name="zksk", attr=zksk_ast.attrs["zksk"])
    lib_ast.add_attr(attr_name="petlib", attr=petlib_ast.attrs["petlib"])
    lib_ast.syft.add_attr("core", attr=plan_ast.syft.core)

    # let the misc creation be always the last, as it needs the full ast solved
    # to etproperly generated unions
    union_misc_ast = getattr(getattr(create_union_ast(lib_ast, client), "syft"), "lib")
    misc_root = getattr(getattr(lib_ast, "syft"), "lib")
    misc_root.add_attr(attr_name="misc", attr=union_misc_ast.attrs["misc"])

    return lib_ast


lib_ast = create_lib_ast(None)
