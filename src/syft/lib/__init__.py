# stdlib
import importlib
from typing import Any as TypeAny
from typing import Dict as TypeDict

# third party
from packaging import version

# syft relative
from ..ast.globals import Globals
from ..lib.python import create_python_ast
from ..lib.torch import create_torch_ast
from ..lib.torchvision import create_torchvision_ast
from .misc import create_union_ast


class VendorLibraryImportException(Exception):
    pass


def vendor_requirements_available(vendor_requirements: TypeDict[str, TypeAny]) -> bool:
    # see if torch version is supported
    if "torch" in vendor_requirements:
        torch_reqs = vendor_requirements["torch"]
        # third party
        import torch

        TORCH_VERSION = version.parse(torch.__version__.split("+")[0])
        min_version = torch_reqs.get("min_version", None)
        if min_version is not None:
            if TORCH_VERSION < version.parse(min_version):
                raise VendorLibraryImportException(
                    f"Unable to load {vendor_requirements['lib']}."
                    + f"Torch: {TORCH_VERSION} < {min_version}"
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
                update_ast(ast=lib_ast)

                for _, client in lib_ast.registered_clients.items():
                    update_ast(ast=client)

                # cache the constructor for future created clients
                lib_ast.loaded_lib_constructors[lib] = update_ast
    except VendorLibraryImportException as e:
        print(e)
    except Exception as e:
        print(f"Unable to load package support for: {lib}. {e}")


# now we need to load the relevant frameworks onto the node
def create_lib_ast() -> Globals:
    lib_ast = Globals()

    python_ast = create_python_ast()
    lib_ast.add_attr(attr_name="syft", attr=python_ast.attrs["syft"])

    torch_ast = create_torch_ast()
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])

    torchvision_ast = create_torchvision_ast()
    lib_ast.add_attr(attr_name="torchvision", attr=torchvision_ast.attrs["torchvision"])

    # let the misc creation be always the last, as it needs the full ast solved
    # to properly generated unions
    misc_ast = getattr(getattr(create_union_ast(lib_ast), "syft"), "lib")
    misc_root = getattr(getattr(lib_ast, "syft"), "lib")

    misc_root.add_attr(attr_name="misc", attr=misc_ast.attrs["misc"])
    return lib_ast


# constructor: copyType = create_lib_ast
lib_ast = create_lib_ast()
lib_ast._copy = create_lib_ast
