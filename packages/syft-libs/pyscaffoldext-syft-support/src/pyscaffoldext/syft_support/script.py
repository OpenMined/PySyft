"""
Running script: python package_support_script.py -l xgboost -d 1
Generates:

1. <LIB>.package_support.json
2. <LIB>.debug.log with all the methods/functions not added to ast with the reason.
3. [optional] also generates <LIB>.return_types.ipynb for the first time.
    If presents, extracts return types from the notebook
"""
# stdlib
import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys
import typing
from os import path
from typing import Any as TypeAny
from typing import Optional, Union
from typing import Dict as TypeDict
from typing import Tuple as TypeTuple
from typing import List as TypeList

import nbformat as nbf
from pyscaffold.operations import no_overwrite

# package_name = 'xgboost'


def list_submodules(list_name: TypeAny, package_name: TypeAny) -> TypeAny:
    try:
        prefix = package_name.__name__
        for loader, module_name, is_pkg in pkgutil.walk_packages(package_name.__path__):

            if "test" in module_name:
                continue
                # inspect.ismodule(__import__('sklearn.neighbors.tests.test_neighbors_tree')) is True
            module_name = f"{prefix}.{module_name}"
            list_name.append(module_name)
            module_name = __import__(module_name, fromlist="dummylist")
            if is_pkg:
                list_submodules(list_name, module_name)
    except Exception as e:
        # print("error with prefix", prefix)
        # e = e

        print(f"list_submodules error:\n package_name = {package_name.__name__}\n{e} ")


def set_classes(
    modules_list: TypeAny,
    root_module: str,
    debug_list: TypeAny,
) -> TypeAny:

    classes_set = set()
    allowlist: TypeList[str, str] = []
    # print(f'Len of modules_list {len(modules_list)}')
    for i in modules_list:
        try:
            module = importlib.import_module(i)
            # print(f'{module} {i}')
            for ax in dir(module):
                # print(ax)
                # print(f' {module.__name__}, {ax}')
                t = getattr(module, ax)
                if inspect.isclass(t):  # classes in modules
                    mod_name = t.__module__.split(".")
                    if root_module == mod_name[0]:
                        # print(f'{t} {t.__module__}')
                        # classes_set.add(module.__name__ + "." + t.__name__) # Number of classes 1224
                        classes_set.add(i + "." + ax)
                        """
                        classes_set.add(
                            t.__module__ + "." + t.__name__
                        )  # for sklearn: number of classes 500

                        """

                    # else:
                    # print(f'in else {t.__name__} {t.__class__} {module} {root_module}')

                # ToDo: add methods/fuctions in modules to allowlist
                # Example: `statsmodels.api.add_constant`
                if (
                    inspect.ismethod(t)
                    or inspect.isfunction(t)
                    or inspect.isgetsetdescriptor(t)
                ):
                    # print(f't for debug: {t} {module}')
                    is_error, string = get_return_type(t, i, ax)
                    if is_error:
                        debug_list.append(string)
                    else:
                        allowlist.append((f"{i}.{t.__name__}", string))
        except Exception as e:
            # print(f"set_classes: module_name = {i}: exception occoured \n\t{e}")
            debug_list.append(
                f"set_classes: module_name = {i}: exception occoured \n\t{e}"
            )

    # print(f'Len of classes_set {len(classes_set)}')
    return classes_set, debug_list, allowlist


def class_import(name: TypeAny) -> TypeAny:
    components = name.split(".")
    mod = importlib.import_module(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp, None)
    return mod


def get_return_type(t: TypeAny, i: str, ax: str) -> TypeAny:
    """
    Argument:
    t: Name of method/function of the class
    Returns:
    is_error: (binary)
        0: add returned string in allowlist
        1: add returned list in debuglist
    """
    try:
        # try block
        d = typing.get_type_hints(t)
        if not d:
            return 0, "_syft_missing"
            # Suggested by @Madhava, enginner can change it
            # Semi-automated
            # return 1, f"{i}.{t.__name__}: type hints absent"
            # debug_list.append(f"{i}.{t.__name__}: type hints absent")
        else:
            if "return" in d.keys():
                if isinstance(d["return"], typing._GenericAlias):  # type: ignore
                    # print(type(d['return']))
                    # print(get_origin(d['return']))
                    """
                    allowlist[i + "." + t.__name__] = get_origin(
                        d["return"]
                    ).__name__
                    """
                    return 0, str(d["return"])
                else:
                    # print(d['return'])
                    if d["return"].__module__ == "builtins":
                        # avoid outputs like 'builtins.str'
                        """
                        allowlist[i + "." + t.__name__] = d[
                            "return"
                        ].__qualname__
                        """
                        return 0, d["return"].__qualname__
                    else:
                        """
                        allowlist[i + "." + t.__name__] = (
                            d["return"].__module__
                            + "."
                            + d["return"].__name__
                        )"""

                        return 0, (d["return"].__module__ + "." + d["return"].__name__)

                # allowlist[module.__name__ + '.' + t.__name__] = d['return'].__name__
            else:

                """
                debug_list.append(
                    f"{i}.{t.__name__}: return key absent in {d}"
                )
                """

                # return 1, f"{i}.{t.__name__}: return key absent in {d}"
                return 0, "_syft_return_absent"
    except Exception as e:
        return 1, f"{i}.{ax}: exception occoured \n\t{e}"
        # debug_list.append(f"{i}.{t.__name__}: exception occoured \n\t{e}")


def dict_allowlist(
    i: TypeAny,
) -> TypeAny:

    methods_error_count = 0
    missing_return = 0
    # for i in classes_set:
    debug_list: TypeList[str] = list()
    allowlist: TypeList[TypeTuple[str, str]] = []
    list_nb: TypeList[TypeAny] = list()
    class_ = class_import(i)
    if_class_added = False
    # print(class_)
    if class_ is None:
        return allowlist, debug_list, methods_error_count, missing_return, list_nb
    for ax in dir(class_):
        # print(f'{ax} {class_}')
        # module = class_
        t = getattr(class_, ax, None)  # Sometimes it return None
        if t is None:
            # print('None')
            continue

        if (
            inspect.ismethod(t)
            or inspect.isfunction(t)
            or inspect.isgetsetdescriptor(t)
            # or isinstance(t, property) # Properties don't have return types :
        ):
            # print(f't for debug: {t} {module}')
            is_error, string = get_return_type(t, i, ax)
            if is_error:
                debug_list.append(string)
                methods_error_count += 1
            else:
                if string in ["_syft_missing", "_syft_return_absent"]:
                    missing_return += 1
                    if not if_class_added:
                        list_nb.append(nbf.v4.new_markdown_cell(f"## {i}"))
                        if_class_added = True

                    i_ = i.replace(".", "_")

                    code = (
                        f"# {i}.{t.__name__}\n"
                        f"try:\n"
                        f"\tobj ={i}()\n"
                        f"\tret = obj.{t.__name__}()\n"
                        f"\ttype_{i_}_{t.__name__} = ret.__module__ + '.' + ret.__class__.__name__\n"
                        f"\tprint('{i}.{t.__name__}: Done')\n"
                        f"except Exception as e:\n"
                        f"\ttype_{i_}_{t.__name__} = '_syft_missing'\n"
                        f"\tprint('{i}.{t.__name__}: Return unavailable')\n"
                        f'\tprint("  Please fix this return type code until there is no exception")\n'
                        f"\tprint('  Error:', e)\n"
                    )

                    list_nb.append(nbf.v4.new_code_cell(code))
                allowlist.append((f"{i}.{t.__name__}", string))
    return allowlist, debug_list, methods_error_count, missing_return, list_nb


def generate_package_support(
    package_name: str, DEBUG: bool = False
) -> TypeTuple[str, TypeDict[str, str]]:

    # DEBUG = args.debug
    # package_name = args.lib

    DEBUG_FILE_NAME = f"{package_name}.debug.log"
    PKG_SUPPORT_NAME = f"{package_name}.pkg_support.json"

    # missing_return_dir holds all the notebooks and files
    # in missing_return/ directory
    # with file_name:content as key:value pair
    missing_returns_dir: TypeDict[str, str] = {}

    # create_nb = True
    # create_nb = False

    #
    # list_nb = []
    # list_nb.append(nbf.v4.new_markdown_cell(f"# {package_name}"))
    # list_nb.append(nbf.v4.new_code_cell(f"import {package_name}"))

    try:
        package = __import__(package_name)
    except ImportError:
        print(f"Package {package_name} not found...")
        sys.exit(1)

    modules_list = [package_name]
    debug_list: TypeList[str] = list()

    list_submodules(modules_list, package)

    # print(f"Number of modules {len(modules_list)}")

    classes_set, debug_list, allowlist = set_classes(
        modules_list, package_name, debug_list
    )

    # print(f"Number of classes {len(classes_set)}")
    """
    allowlist, debug_list, methods_error_count, missing_return = dict_allowlist_original(
        classes_set, debug_list, allowlist, list_nb
    )
    """

    # Restructuring dict_allowlist
    methods_error_count = 0
    missing_return = 0
    list_nb: TypeList[TypeAny] = list()
    missing_classes = list()
    for class_ in classes_set:
        (
            allowlist_i,
            debug_list_i,
            methods_error_count_i,
            missing_return_i,
            list_nb_i,
        ) = dict_allowlist(class_)
        allowlist += allowlist_i  # append to list

        debug_list.extend(debug_list_i)

        if len(list_nb_i) > 0:
            nb = nbf.v4.new_notebook()
            class_name = class_.replace(".", "_")
            nb_name = f"{class_name}.ipynb"

            missing_classes.append(class_name)

            nb["cells"] = list_nb_i
            missing_returns_dir[nb_name] = (nbf.writes(nb), no_overwrite())

            list_nb.extend(list_nb_i)
        methods_error_count += methods_error_count_i
        missing_return += missing_return_i

    # print(f"len(allowlist) = {len(allowlist)}")

    if len(missing_classes) > 0:
        # create an __init__ file :)
        init_file = "".join(f"from . import {a}\n" for a in missing_classes)
        missing_returns_dir["__init__.py"] = (init_file, no_overwrite())

    package_support: TypeDict[str, TypeAny] = dict()

    package_support["lib"] = package_name
    # petlib doesnot have version
    # package_support["Version"] = package.__version__
    # sort it all
    classlist = list(classes_set)
    classlist.sort()
    modules_list.sort()
    allowlist.sort()

    package_support["classes"] = classlist
    package_support["modules"] = modules_list
    package_support["methods"] = allowlist

    print(f"-----{package_name} Summary-----")
    print("Modules")
    print(f"\tAdded:{len(modules_list)}")
    print("Classes")
    print(f"\tAdded:{len(classes_set)}")
    print("Methods")
    print(f"\tAdded:{len(allowlist) - missing_return}")
    print(f"\tReturn type absent: {missing_return}")
    print(f"\tNot added:{methods_error_count}")
    print("-----------------")

    return json.dumps(package_support), missing_returns_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", dest="lib", required=True, help="name of the model to be added to ast"
    )
    parser.add_argument(
        "-d", dest="debug", type=int, help="Set it to one to get debug files", default=0
    )
    args = parser.parse_args()

    DEBUG = args.debug
    package_name = args.lib

    generate_package_support(package_name, DEBUG)
