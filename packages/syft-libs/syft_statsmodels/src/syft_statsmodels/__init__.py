# stdlib
import glob
import json
import os
from importlib import import_module
from os.path import basename
from typing import Any as TypeAny
from typing import Dict, Iterable
from typing import List as TypeList
from typing import Tuple as TypeTuple


def read_package_support() -> TypeList[TypeTuple[str, TypeAny]]:
    with open(
        os.path.join(os.path.dirname(__file__), "package-support.json"), "r"
    ) as f:
        data = json.load(f)

    # TODO: check type to prevent errors and raise necessary errors
    modules = [(t[0], import_module(t[1])) for t in data["modules"] if len(t) == 2]
    classes = []
    for t in data["classes"]:
        module, classname = t[2].rsplit(".", 1)
        klass = getattr(import_module(module), classname)
        classes.append((t[0], t[1], klass))

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

        if isinstance(serde, Iterable) and not isinstance(serde, Dict):
            serde_objs.extend(serde)
        else:
            serde_objs.append(serde)

    return serde_objs


config = read_package_support()
objects = get_serde()
