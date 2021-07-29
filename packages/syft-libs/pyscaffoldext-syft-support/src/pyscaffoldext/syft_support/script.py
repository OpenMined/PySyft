"""
Running script: python package_support_script.py -l xgboost -d 1
Generates:

1. package_support.json
2. lib.debug.log with all the methods/functions not added to ast with the reason.
"""
# stdlib
import argparse
import importlib
import inspect
import json
import pkgutil
import sys
import typing
from typing import Any as TypeAny

parser = argparse.ArgumentParser()
parser.add_argument(
    "-l", dest="lib", required=True, help="name of the model to be added to ast"
)
parser.add_argument(
    "-d", dest="debug", type=int, help="Set it to one to get debug files", default=0
)
args = parser.parse_args()

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
    allowlist = {}
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
                    is_error, string = get_return_type(t, i)
                    if is_error:
                        debug_list.append(string)
                    else:
                        allowlist[i + "." + t.__name__] = string
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


def get_return_type(t: TypeAny, i: str) -> TypeAny:
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
        return 1, f"{i}.{t.__name__}: exception occoured \n\t{e}"
        # debug_list.append(f"{i}.{t.__name__}: exception occoured \n\t{e}")


def dict_allowlist(
    classes_set: TypeAny, debug_list: TypeAny, allowlist: dict
) -> TypeAny:

    # allowlist = {}
    methods_error_count = 0
    missing_return = 0
    for i in classes_set:
        class_ = class_import(i)
        # print(class_)
        if class_ is None:
            continue
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
                or isinstance(t, property)
            ):
                # print(f't for debug: {t} {module}')
                is_error, string = get_return_type(t, i)
                if is_error:
                    debug_list.append(string)
                    methods_error_count += 1
                else:
                    if string in ["_syft_missing", "_syft_return_absent"]:
                        missing_return += 1
                    allowlist[i + "." + t.__name__] = string
    return allowlist, debug_list, methods_error_count, missing_return


def main() -> None:

    DEBUG = args.debug
    package_name = args.lib

    DEBUG_FILE_NAME = f"{package_name}.debug.log"
    PKG_SUPPORT_NAME = f"{package_name}.pkg_support.json"

    try:
        package = __import__(package_name)
    except ImportError:
        print(f"Package {package_name} not found...")
        sys.exit(1)

    modules_list = [package_name]
    debug_list = list()  # type: ignore

    list_submodules(modules_list, package)

    # print(f"Number of modules {len(modules_list)}")

    classes_set, debug_list, allowlist = set_classes(
        modules_list, package_name, debug_list
    )

    # print(f"Number of classes {len(classes_set)}")

    allowlist, debug_list, methods_error_count, missing_return = dict_allowlist(
        classes_set, debug_list, allowlist
    )

    # print(f"len(allowlist) = {len(allowlist)}")
    package_support = {}

    package_support["lib"] = package_name
    # petlib doesnot have version
    # package_support["Version"] = package.__version__
    package_support["class"] = list(classes_set)
    package_support["modules"] = modules_list
    package_support["methods"] = allowlist

    with open(PKG_SUPPORT_NAME, "w") as outfile:
        json.dump(package_support, outfile)

    if DEBUG:
        # print(debug_list)
        with open(DEBUG_FILE_NAME, "w") as f:
            for item in debug_list:
                f.write(f"{item}\n")

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


if __name__ == "__main__":
    main()
