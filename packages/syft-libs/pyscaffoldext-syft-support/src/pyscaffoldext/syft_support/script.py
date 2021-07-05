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
from types import ModuleType
from typing import Any as TypeAny
from typing import Dict
from typing import List as TypeList
from typing import Optional, Union

# third party
from typing_inspect import get_origin


def list_submodules(list_name: TypeAny, package_name: TypeAny) -> TypeAny:
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        package_name.__path__, package_name.__name__ + "."
    ):
        if "test" in module_name:
            continue
            # inspect.ismodule(__import__('sklearn.neighbors.tests.test_neighbors_tree')) is True
        list_name.append(module_name)
        try:
            module_name = __import__(module_name, fromlist="dummylist")
        except Exception as e:
            print(e)
        if is_pkg:
            list_submodules(list_name, module_name)


def set_classes(modules_list: TypeAny, root_module: str) -> TypeAny:

    classes_set = set()
    # print(f'Len of modules_list {len(modules_list)}')
    for i in modules_list:
        module = importlib.import_module(i)
        # print(f'{module} {i}')
        for ax in dir(module):
            # print(ax)
            # print(f' {module.__name__}, {ax}')
            t = getattr(module, ax)
            if inspect.isclass(t):
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
                    if t.__module__ == "xgboost.core":
                        print(
                            f'{t.__module__ + "." + t.__name__} and {module.__name__ + "." + t.__name__}'
                        )

                # else:
                # print(f'in else {t.__name__} {t.__class__} {module} {root_module}')

    # print(f'Len of classes_set {len(classes_set)}')
    return classes_set


def class_import(name: TypeAny) -> TypeAny:
    components = name.split(".")
    mod = importlib.import_module(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


Primitive_types_mapping: Dict[str, str] = {
    "int": "syft.lib.python.Int",
    "float": "syft.lib.python.Float",
    "NoneType": "syft.lib.python._SyNone",
    "str": "syft.lib.python.String",
    "tuple": "syft.lib.python.Tuple",
    "bool": "syft.lib.python.Bool",
    "list": "syft.lib.python.List",
    "dict": "syft.lib.python.Dict",
}


def update_primite_type(return_type: str) -> str:
    """Return the syft return type for primite types"""
    if return_type in Primitive_types_mapping:
        return Primitive_types_mapping[return_type]
    else:
        return return_type


def dict_allowlist(classes_list: TypeAny) -> TypeAny:

    allowlist = []
    debug_list = list()
    for i in classes_list:
        class_ = class_import(i)
        # print(class_)
        for ax in dir(class_):
            # print(f'{ax} {class_}')
            # module = class_
            t = getattr(class_, ax)  # Sometimes it return None
            if t is None:
                # print('None')
                continue
            # TODO: add support for properties
            if inspect.ismethod(t) or inspect.isfunction(t):
                # print(f't for debug: {t} {module}')
                try:
                    # try block
                    d = typing.get_type_hints(t)
                    if not d:
                        debug_list.append(f"{i}.{t.__name__}: type hints absent")
                    else:
                        if "return" in d.keys():
                            if isinstance(d["return"], typing._GenericAlias):  # type: ignore
                                # print(type(d['return']))
                                # print(get_origin(d['return']))
                                allowlist.append(
                                    (
                                        f"{i}.{t.__name__}",
                                        update_primite_type(
                                            get_origin(d["return"]).__name__
                                        ),
                                    )
                                )
                            else:
                                # print(d['return'])
                                if d["return"].__module__ == "builtins":
                                    # avoid outputs like 'builtins.str'
                                    allowlist.append(
                                        (
                                            f"{i}.{t.__name__}",
                                            update_primite_type(
                                                d["return"].__qualname__
                                            ),
                                        )
                                    )
                                else:
                                    allowlist.append(
                                        (
                                            f"{i}.{t.__name__}",
                                            "{}.{}".format(
                                                d["return"].__module__,
                                                d["return"].__name__,
                                            ),
                                        )
                                    )

                            # allowlist[module.__name__ + '.' + t.__name__] = d['return'].__name__
                        else:
                            debug_list.append(
                                f"{i}.{t.__name__}: return key absent in {d}"
                            )

                except Exception as e:
                    debug_list.append(f"{i}.{t.__name__}: exception occoured \n\t{e}")

    return allowlist, debug_list


def generate_package_support(
    package: ModuleType, output_file: Optional[str] = None, debug: bool = False
) -> Optional[str]:
    package_name = package.__name__
    modules_list = [package_name]

    list_submodules(modules_list, package)

    classes_list = list(set_classes(modules_list, package_name))

    print(f"Number of classes {len(classes_list)}")

    debug_list = []
    allowlist, debug_list = dict_allowlist(classes_list)

    print(f"len(allowlist) = {len(allowlist)}")
    package_support: Dict[str, Union[str, TypeList[TypeAny]]] = {}

    package_support["lib"] = package_name
    package_support["classes"] = classes_list
    package_support["modules"] = modules_list
    package_support["methods"] = allowlist

    if output_file:
        with open("package_support.json", "w") as outfile:
            json.dump(package_support, outfile)
    else:
        return json.dumps(package_support)

    if debug:
        # print(debug_list)
        with open("lib.debug.log", "w") as f:
            for item in debug_list:
                f.write(f"{item}\n")
    return  # type: ignore


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", dest="lib", required=True, help="name of the model to be added to ast"
    )
    parser.add_argument(
        "-d", dest="debug", type=int, help="Set it to one to get debug files", default=0
    )
    parser.add_argument(
        "-o",
        dest="output",
        type=str,
        help="name of the output file",
        default="package-support.json",
    )

    args = parser.parse_args()

    DEBUG = args.debug
    package_name = args.lib
    output_file = args.output
    try:
        package = __import__(package_name)
    except ImportError:
        print(f"Package {package_name} not found...")
        sys.exit(1)

    generate_package_support(package, output_file=output_file, debug=DEBUG)
