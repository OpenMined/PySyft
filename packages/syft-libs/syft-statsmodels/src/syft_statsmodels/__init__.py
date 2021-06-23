# stdlib
import glob
import inspect
import json
import os
from importlib import import_module
from os.path import basename
from typing import Any as TypeAny
from typing import Dict, Iterable
from typing import List as TypeList

# syft absolute
import syft as sy


def read_package_support() -> Dict[str, TypeAny]:
    with open(
        os.path.join(os.path.dirname(__file__), "package-support.json"), "r"
    ) as f:
        data = json.load(f)

    modules = [
        (module_name, import_module(module_name)) for module_name in data["modules"]
    ]
    classes = []
    for path in data["classes"]:
        module, classname = path.rsplit(".", 1)
        klass = getattr(import_module(module), classname)

        if not inspect.isclass(klass):
            raise TypeError(f"{path} is not a class.")
        classes.append((path, path, klass))

    return {
        "lib": data["lib"],
        "modules": modules,
        "classes": classes,
        "methods": data["methods"],
    }


def get_serde() -> TypeList[Dict[str, TypeAny]]:
    serde_objs: TypeList[Dict[str, TypeAny]] = []
    dir_path = os.path.dirname(__file__)
    _, dir_name = dir_path.rsplit("/", 1)

    all_serde_modules = glob.glob(os.path.join(dir_path, "serde/*.py"))
    for f in all_serde_modules:
        module_path = "{}.serde.{}".format(dir_name, basename(f)[:-3])
        serde_module = import_module(module_path)
        serde = getattr(serde_module, "serde")
        if isinstance(serde, Iterable) and not isinstance(serde, Dict):
            serde_objs.extend(serde)
        else:
            serde_objs.append(serde)

    return serde_objs


sy.lib.add_lib_external(read_package_support(), get_serde())
