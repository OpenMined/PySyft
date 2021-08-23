# stdlib
import functools
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
import packaging
from packaging import version
import wrapt

# relative
from ..ast import add_classes
from ..ast import add_methods
from ..ast import add_modules
from ..ast.globals import Globals
from ..core.node.abstract.node import AbstractNodeClient
from ..core.tensor import create_tensor_ast
from ..generate_wrapper import GenerateWrapper
from ..lib.plan import create_plan_ast
from ..lib.python import create_python_ast
from ..lib.remote_dataloader import create_remote_dataloader_ast
from ..lib.torch import create_torch_ast
from ..lib.torchvision import create_torchvision_ast
from ..logger import critical
from ..logger import traceback_and_raise
from ..logger import warning
from .misc import create_union_ast
from .misc.union import UnionGenerator
from .util import generic_update_ast


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
    ignore_warning: bool = False,
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
    # Note: syft.load(lib = "opacus") doesnot work as it iterates the string, syft.load('opacus') works

    if not ignore_warning:
        msg = "sy.load() is deprecated and not needed anymore"
        warning(msg, print=True)
        warnings.warn(msg, DeprecationWarning)

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
    _load_lib() is deprecated please use load() in the future

    Args:
        lib: name of library to load and update Node with
        options: external requirements for loading library successfully

    """
    msg = "sy._load_lib() is deprecated and not needed anymore"
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
    tensor_ast = create_tensor_ast(client=client)

    lib_ast = Globals(client=client)
    lib_ast.add_attr(attr_name="syft", attr=python_ast.attrs["syft"])
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])
    lib_ast.add_attr(attr_name="torchvision", attr=torchvision_ast.attrs["torchvision"])
    lib_ast.syft.add_attr("core", attr=plan_ast.syft.core)
    lib_ast.syft.core.add_attr("tensor", attr=tensor_ast.syft.core.tensor)
    lib_ast.syft.core.add_attr(
        "remote_dataloader", remote_dataloader_ast.syft.core.remote_dataloader
    )

    # let the misc creation be always the last, as it needs the full ast solved
    # to properly generated unions
    union_misc_ast = getattr(getattr(create_union_ast(lib_ast, client), "syft"), "lib")
    lib_ast.syft.lib.add_attr(attr_name="misc", attr=union_misc_ast.attrs["misc"])

    return lib_ast


lib_ast = create_lib_ast(None)


def register_library(lib: str) -> None:
    """
    Add newly imported library to shared lib space.
    """
    package_support_path = "__SYFT_PACKAGE_SUPPORT"
    # load(lib, ignore_warning=True)
    if package_support_path not in sys.modules:
        sys.modules[package_support_path] = set()
    sys.modules[package_support_path].add(lib)
    # bind_library(lib)


@wrapt.when_imported("gym")
@wrapt.when_imported("opacus")
@wrapt.when_imported("numpy")
@wrapt.when_imported("sklearn")
@wrapt.when_imported("pandas")
@wrapt.when_imported("PIL")
@wrapt.when_imported("petlib")
@wrapt.when_imported("openmined_psi")
@wrapt.when_imported("pydp")
@wrapt.when_imported("statsmodels")
@wrapt.when_imported("sympc")
@wrapt.when_imported("tenseal")
# @wrapt.when_imported("xgboost")
@wrapt.when_imported("zksk")
@wrapt.when_imported("pytorch_lightning")
@wrapt.when_imported("transformers")
def post_import_hook_third_party(module: TypeAny) -> None:
    """
    Note: This needs to be after `lib_ast` because code above uses lib-ast
    """
    # msg = f"inside post_import_hook_third_party module_name {module.__name__}"
    # warning(msg, print=True)
    # warnings.warn(msg, DeprecationWarning)
    load(module.__name__, ignore_warning=True)
    register_library(module.__name__)


def support_packages() -> TypeSet:
    """
    Returns a set of all syft packages currently imported.
    """
    return sys.modules["__SYFT_PACKAGE_SUPPORT"]


def _map2syft_types(
    methods: TypeList[TypeTuple[str, str]]
) -> TypeList[TypeTuple[str, str]]:
    primitive_map = {
        "bool": "syft.lib.python.Bool",
        "complex": "syft.lib.python.Complex",
        "dict": "syft.lib.python.Dict",
        "float": "syft.lib.python.Float",
        "int": "syft.lib.python.Int",
        # "Iterator":"syft.lib.python.Iterator",
        "list": "syft.lib.python.List",
        "NoneType": "syft.lib.python._SyNone",
        "range": "syft.lib.python.Range",
        "set": "syft.lib.python.Set",
        "slice": "syft.lib.python.Slice",
        "str": "syft.lib.python.String",
        "tuple": "syft.lib.python.Tuple",
    }
    for i, (func, return_type) in enumerate(methods):
        if return_type.startswith("Union"):
            types = return_type[5:].strip("[]").split(",")
            for i in range(len(types)):
                if types[i] in primitive_map:
                    types[i] = primitive_map[types[i]]
            methods[i] = (func, UnionGenerator[tuple(types)])

        elif return_type in primitive_map:
            methods[i] = (func, primitive_map[return_type])
    return methods


def _create_support_ast(
    modules: TypeList[TypeTuple[str, TypeAny]],
    classes: TypeList[TypeTuple[str, str, TypeAny]],
    methods: TypeList[TypeTuple[str, str]],
    client: TypeAny = None,
) -> Globals:
    ast = Globals(client=client)
    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()
    return ast


def add_lib_external(
    config: TypeDict[str, TypeAny], objects: Iterable[TypeDict[str, TypeAny]]
) -> None:
    lib = config["lib"]

    # Generate proto wrappers
    if isinstance(objects, Iterable):
        for serde_object in objects:
            GenerateWrapper(**serde_object)
    else:
        critical("Serde objects is expected to be an Iterable.")

    methods = _map2syft_types(config["methods"])
    # create_ast and update_ast function
    create_ast = functools.partial(
        _create_support_ast, config["modules"], config["classes"], methods
    )
    update_ast = functools.partial(generic_update_ast, lib, create_ast)

    # update and add lib to lib_ast and clients
    global lib_ast
    update_ast(ast_or_client=lib_ast)
    # cache the constructor for future created clients
    lib_ast.loaded_lib_constructors[lib] = update_ast
    _regenerate_unions(lib_ast=lib_ast)

    for _, client in lib_ast.registered_clients.items():
        update_ast(ast_or_client=client)
        _regenerate_unions(lib_ast=lib_ast, client=client)
