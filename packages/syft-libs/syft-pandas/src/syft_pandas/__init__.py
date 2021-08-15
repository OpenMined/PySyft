# stdlib
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Iterable
from typing import List as TypeList
from typing import Tuple as TypeTuple

# syft absolute
# absolute
import syft as sy

# relative
from . import serde  # noqa: F401


def read_package_support() -> TypeDict[str, TypeList[TypeTuple[TypeAny, ...]]]:
    with open(
        os.path.join(os.path.dirname(__file__), "package-support.json"), "r"
    ) as f:
        data = json.load(f)

    modules: TypeList[TypeTuple[TypeAny, ...]] = [
        (module_name, import_module(module_name)) for module_name in data["modules"]
    ]
    classes: TypeList[TypeTuple[TypeAny, ...]] = []
    for path in data["classes"]:
        if isinstance(path, list):
            if not len(path) == 2:
                raise ValueError(
                    "Error at {}.\nUse [PATH,RETURN_TYPE] or PATH to specify a class.".format(
                        path
                    )
                )
            path, return_type = path
        else:
            return_type = path
        module, classname = return_type.rsplit(".", 1)
        klass = getattr(import_module(module), classname)

        if not inspect.isclass(klass):
            raise TypeError(f"{path} is not a class.")
        classes.append((path, return_type, klass))

    return {
        "lib": data["lib"],
        "modules": modules,
        "classes": classes,
        "methods": data["methods"],
    }


def get_serde() -> TypeList[TypeDict[str, TypeAny]]:
    serde_objs: TypeList[TypeDict[str, TypeAny]] = []

    dir_path = Path(os.path.dirname(__file__))
    serde_dir = dir_path / "serde"
    if serde_dir.exists():
        for f in serde_dir.iterdir():
            if f.name.endswith(".py"):
                module_path = f"{f.parent.parent.stem}.serde.{f.stem}"
                serde_module = import_module(module_path)
                try:
                    serde_obj = getattr(serde_module, "serde")
                except AttributeError:
                    continue

                if isinstance(serde_obj, Iterable) and not isinstance(serde_obj, dict):
                    serde_objs.extend(serde_obj)
                else:
                    serde_objs.append(serde_obj)

    return serde_objs


sy.lib.add_lib_external(read_package_support(), get_serde())
