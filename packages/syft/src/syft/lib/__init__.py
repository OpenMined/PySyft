# stdlib
import importlib
import sys
from types import ModuleType
from typing import Any
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Iterable
from typing import List as TypeList
from typing import Optional
from typing import Set as TypeSet
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion
import warnings

# third party
from cachetools import cached
from cachetools.keys import hashkey
from packaging import version

# syft relative
from ..ast.globals import Globals
from ..core.node.abstract.node import AbstractNodeClient
from ..lib.plan import create_plan_ast
from ..lib.python import create_python_ast
from ..lib.remote_dataloader import create_remote_dataloader_ast
from ..lib.torch import create_torch_ast
from ..lib.torchvision import create_torchvision_ast
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


def _add_lib(
    *, vendor_ast: ModuleType, ast_or_client: TypeUnion[Globals, AbstractNodeClient]
) -> None:
    update_ast = getattr(vendor_ast, "update_ast", None)
    if update_ast is not None:
        update_ast(ast_or_client=ast_or_client)


def _regenerate_unions(*, lib_ast: Globals, client: TypeAny = None) -> None:
    union_misc_ast = getattr(
        getattr(create_union_ast(lib_ast=lib_ast, client=client), "syft"), "lib"
    )
    if client is not None:
        client.syft.lib.add_attr(attr_name="misc", attr=union_misc_ast.attrs["misc"])
    else:
        lib_ast.syft.lib.add_attr(attr_name="misc", attr=union_misc_ast.attrs["misc"])


@cached({}, lambda *, lib, options=None: hashkey(lib))
def _load_lib(*, lib: str, options: Optional[TypeDict[str, TypeAny]] = None) -> None:
    """
    Load and Update Node with given library module

    Args:
        lib: name of library to load and update Node with
        options: external requirements for loading library successfully
    """
    global lib_ast
    _options = {} if options is None else options

    _ = importlib.import_module(lib)
    vendor_ast = importlib.import_module(f"syft.lib.{lib}")
    PACKAGE_SUPPORT = getattr(vendor_ast, "PACKAGE_SUPPORT", None)
    PACKAGE_SUPPORT.update(_options)
    if PACKAGE_SUPPORT is not None and vendor_requirements_available(
        vendor_requirements=PACKAGE_SUPPORT
    ):
        _add_lib(vendor_ast=vendor_ast, ast_or_client=lib_ast)
        # cache the constructor for future created clients
        lib_ast.loaded_lib_constructors[lib] = getattr(vendor_ast, "update_ast", None)
        _regenerate_unions(lib_ast=lib_ast)

        for _, client in lib_ast.registered_clients.items():
            _add_lib(vendor_ast=vendor_ast, ast_or_client=client)
            _regenerate_unions(lib_ast=lib_ast, client=client)


def load(
    *libs: TypeUnion[TypeList[str], TypeTuple[str], TypeSet[str], str],
    options: TypeDict[str, TypeAny] = {},
    **kwargs: str,
) -> None:
    """
    Load and Update Node with given library module

    Args:
        *libs: names of libraries to load and update Node with (can be variadic, tuple, list, set)
        options: external requirements for loading library successfully
        **kwargs: for backward compatibility with calls like `syft.load(lib = "opacus")`
    """
    # For backward compatibility with calls like `syft.load(lib = "opacus")`
    if "lib" in kwargs.keys():
        libs += tuple(kwargs["lib"])

    if isinstance(libs[0], Iterable):
        if not isinstance(libs[0], str):
            libs = tuple(libs[0])
        for lib in libs:
            if isinstance(lib, str):
                try:
                    _load_lib(lib=str(lib), options=options)
                except VendorLibraryImportException as e:
                    critical(e)
                except Exception as e:
                    critical(f"Unable to load package support for: {lib}. {e}")
            else:
                critical(
                    f"Unable to load package support for: {lib}. Pass lib name as string object."
                )
    else:
        critical(
            "Unable to load package support for any library. Iterable object not found."
        )


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
    # numpy_ast = create_numpy_ast()
    plan_ast = create_plan_ast(client=client)
    remote_dataloader_ast = create_remote_dataloader_ast(client=client)

    lib_ast = Globals(client=client)
    lib_ast.add_attr(attr_name="syft", attr=python_ast.attrs["syft"])
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])
    lib_ast.add_attr(attr_name="torchvision", attr=torchvision_ast.attrs["torchvision"])
    lib_ast.syft.add_attr("core", attr=plan_ast.syft.core)
    lib_ast.syft.core.add_attr(
        "remote_dataloader", remote_dataloader_ast.syft.core.remote_dataloader
    )

    # let the misc creation be always the last, as it needs the full ast solved
    # to properly generated unions
    union_misc_ast = getattr(getattr(create_union_ast(lib_ast, client), "syft"), "lib")
    lib_ast.syft.lib.add_attr(attr_name="misc", attr=union_misc_ast.attrs["misc"])

    return lib_ast


lib_ast = create_lib_ast(None)
