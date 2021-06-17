# stdlib
import glob
from importlib import import_module
import inspect
import json
import os
from os.path import basename
from typing import Any as TypeAny
from typing import Dict
from typing import Iterable
from typing import List as TypeList
from typing import Tuple as TypeTuple

# syft absolute
import syft as sy


def read_package_support() -> TypeList[TypeTuple[str, TypeAny]]:
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

    # TODO: can we test if methods are correct?
    return {
        "lib": data["lib"],
        "modules": modules,
        "classes": classes,
        "methods": data["methods"],
    }


def get_serde() -> TypeList[Dict[str, TypeAny]]:
    serde_objs = []
    dir_path = os.path.dirname(__file__)
    _, dir_name = dir_path.rsplit("/", 1)

    all_serde_modules = glob.glob(os.path.join(dir_path, "serde/*.py"))
    for f in all_serde_modules:
        module_path = "{}.serde.{}".format(dir_name, basename(f)[:-3])
        serde_module = import_module(module_path)
        serde = getattr(serde_module, "serde")
        # TODO: check serde
        if isinstance(serde, Iterable) and not isinstance(serde, Dict):
            serde_objs.extend(serde)
        else:
            serde_objs.append(serde)

    return serde_objs


sy.lib.add_lib_external(read_package_support(), get_serde())
