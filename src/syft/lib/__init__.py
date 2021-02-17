# stdlib
import importlib
import sys
from typing import Any
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Optional

# third party
from packaging import version

# syft relative
from ..ast.globals import Globals
from ..lib.action import create_plan_ast
from ..lib.python import create_python_ast
from ..lib.torch import create_torch_ast
from ..lib.torchvision import create_torchvision_ast
from ..logger import critical
from ..logger import traceback_and_raise
from .misc import create_union_ast


class VendorLibraryImportException(Exception):
    pass


def vendor_requirements_available(vendor_requirements: TypeDict[str, TypeAny]) -> bool:
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

    return True


def load_lib(lib: str, options: TypeDict[str, TypeAny] = {}) -> None:
    try:
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
    except VendorLibraryImportException as e:
        critical(e)
    except Exception as e:
        critical(f"Unable to load package support for: {lib}. {e}")


# now we need to load the relevant frameworks onto the node
def create_lib_ast(client: Optional[Any] = None) -> Globals:
    python_ast = create_python_ast(client=client)
    torch_ast = create_torch_ast(client=client)
    torchvision_ast = create_torchvision_ast(client=client)
    # numpy_ast = create_numpy_ast()
    plan_ast = create_plan_ast(client=client)

    lib_ast = Globals(client=client)
    lib_ast.add_attr(attr_name="syft", attr=python_ast.attrs["syft"])
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])
    lib_ast.add_attr(attr_name="torchvision", attr=torchvision_ast.attrs["torchvision"])
    lib_ast.syft.add_attr("core", attr=plan_ast.syft.core)

    # let the misc creation be always the last, as it needs the full ast solved
    # to properly generated unions
    union_misc_ast = getattr(getattr(create_union_ast(lib_ast, client), "syft"), "lib")
    misc_root = getattr(getattr(lib_ast, "syft"), "lib")
    misc_root.add_attr(attr_name="misc", attr=union_misc_ast.attrs["misc"])

    return lib_ast


lib_ast = create_lib_ast(None)
