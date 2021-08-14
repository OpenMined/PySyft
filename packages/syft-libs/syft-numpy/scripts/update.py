# import argparse
# stdlib
import importlib
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
from typing import Any as TypeAny


def class_import(name: TypeAny) -> TypeAny:
    components = name.split(".")
    mod = importlib.import_module(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp, None)
    return mod


def update_json() -> None:
    root_dir = os.path.abspath(Path(os.path.dirname(__file__)) / "..")
    PKG_SUPPORT_NAME = Path(f"{root_dir}", "src", "syft_numpy", "package-support.json")
    path_to_missing = Path(f"{root_dir}", "_missing_return", "__init__.py")
    _missing_return = SourceFileLoader(
        "_missing_return", str(path_to_missing)
    ).load_module()

    with open(PKG_SUPPORT_NAME) as f:
        package_support = json.load(f)

    allowlist = package_support["methods"]

    for x, (method, return_type) in enumerate(allowlist):
        if return_type in ["_syft_missing", "_syft_return_absent"]:

            tmp_class = class_import(".".join(method.split(".")[:-1]))
            original_class = tmp_class.__module__ + "." + tmp_class.__name__
            class_ = original_class.replace(".", "_")

            method_name = method.split(".")[-1]
            method_path = original_class + "." + method_name
            method_ = method_path.replace(".", "_")

            # executing this string should work :)
            try:
                return_type = eval(f"_missing_return.{class_}.type_{method_}")
                if return_type not in ["_syft_missing", "_syft_return_absent"]:
                    print(f"Updating {return_type}")
                    allowlist[x] = (method, return_type)
            except Exception as e:
                print(f"Some Exception in update.py\n\t{e}")

    package_support["methods"] = allowlist

    with open(PKG_SUPPORT_NAME, "w") as outfile:
        json.dump(package_support, outfile)


if __name__ == "__main__":
    update_json()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-l", dest="lib", required=True, help="name of the model to be added to ast"
#     )
#     args = parser.parse_args()

#     package_name = args.lib
#     update_json(package_name)
