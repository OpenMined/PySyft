# stdlib
import argparse
import inspect
import json
import pkgutil
import sys
import typing
from typing import Any as TypeAny

# third party
from typing_inspect import get_origin

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", dest="debug", type=int, help="Set it to one to get debug files", default=0
)
parser.add_argument(
    "-l", dest="lib", required=True, help="name of the model to be added to ast"
)
args = parser.parse_args()

# package_name = 'xgboost'


def list_submodules(list_name: TypeAny, package_name: TypeAny) -> TypeAny:
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        package_name.__path__, package_name.__name__ + "."
    ):
        list_name.append(module_name)
        try:
            module_name = __import__(module_name, fromlist="dummylist")
        except Exception as e:
            print(e)
        if is_pkg:
            list_submodules(list_name, module_name)


def list_classes(modules_list: TypeAny) -> TypeAny:

    classes_list = list()

    for module in modules_list:
        module = __import__(module)
        for ax in dir(module):
            # print(ax)
            # print(f' {module.__name__}, {ax}')
            t = getattr(module, ax)
            if inspect.isclass(t):
                classes_list.append(module.__name__ + "." + t.__name__)

    return classes_list


def class_import(name: TypeAny) -> TypeAny:
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def dict_allowlist(classes_list: TypeAny) -> TypeAny:

    allowlist = {}
    debug_list = list()
    for i in classes_list:
        class_ = class_import(i)
        # print(class_)
        for ax in dir(class_):
            # print(f'{ax} {class_}')
            # module = class_
            t = getattr(class_, ax)
            if inspect.ismethod(t) or inspect.isfunction(t):
                # print(f't for debug: {t} {module}')
                try:
                    # try block
                    d = typing.get_type_hints(t)
                    if not d:
                        debug_list.append(
                            f"{class_.__name__}.{t.__name__}: type hints absent"
                        )
                    else:
                        if "return" in d.keys():
                            if isinstance(d["return"], typing._GenericAlias):  # type: ignore
                                # print(type(d['return']))
                                # print(get_origin(d['return']))
                                allowlist[
                                    class_.__name__ + "." + t.__name__
                                ] = get_origin(d["return"]).__name__
                            else:
                                # print(d['return'])
                                allowlist[class_.__name__ + "." + t.__name__] = d[
                                    "return"
                                ].__name__

                            # allowlist[module.__name__ + '.' + t.__name__] = d['return'].__name__
                        else:
                            debug_list.append(
                                f"{class_.__name__}.{t.__name__}: return key absent in {d}"
                            )

                except Exception as e:
                    debug_list.append(
                        f"{class_.__name__}.{t.__name__}: exception occoured /n/t{e}"
                    )

    return allowlist, debug_list


def main() -> None:

    DEBUG = args.debug
    package_name = args.lib

    try:
        package = __import__(package_name)
    except ImportError:
        print(f"Package {package_name} not found...")
        sys.exit(1)

    modules_list = []  # type: ignore

    list_submodules(modules_list, package)

    classes_list = list_classes(modules_list)

    allowlist, debug_list = dict_allowlist(classes_list)

    package_support = {}

    package_support["lib"] = package_name
    package_support["class"] = classes_list
    package_support["modules"] = modules_list
    package_support["methods"] = allowlist

    with open("package_support.json", "w") as outfile:
        json.dump(package_support, outfile)

    if DEBUG:
        print(debug_list)


if __name__ == "__main__":
    main()
