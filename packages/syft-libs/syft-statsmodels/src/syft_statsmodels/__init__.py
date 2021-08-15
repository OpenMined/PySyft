# stdlib
import inspect
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Iterable
from typing import List as TypeList
from typing import Tuple as TypeTuple

# absolute
# syft absolute
import syft as sy


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
                    serde = getattr(serde_module, "serde")
                except AttributeError:
                    print(f"WARN: No serde found in {module_path}")
                    pass

                if isinstance(serde, Iterable) and not isinstance(serde, dict):
                    serde_objs.extend(serde)
                else:
                    serde_objs.append(serde)

    return serde_objs


sy.lib.add_lib_external(read_package_support(), get_serde())
