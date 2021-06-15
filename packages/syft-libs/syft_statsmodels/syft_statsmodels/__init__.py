# stdlib
import functools
from posixpath import basename, dirname
from typing import Any as TypeAny, Dict, Iterable,Tuple as TypeTuple
from typing import List as TypeList
import json
import glob
import os
from importlib import import_module

# third party
import statsmodels
import statsmodels.api as sm

# syft relative
from .serde import family  # noqa: 401
from .serde import results  # noqa: 401
from syft.ast import add_classes
from syft.ast import add_methods
from syft.ast import add_modules
from syft.ast.globals import Globals
from syft.lib.util import generic_update_ast

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

def get_serde()->TypeList[Dict[str,TypeAny]]: 
    serde_objs = []
    all_serde_modules = glob.glob(os.path.join(os.path.dirname(__file__),"serde/*.py"))     
    for f in all_serde_modules:
        serde_module = import_module("syft_statsmodels.serde."+basename(f)[:-3])
        serde = getattr(serde_module,"serde")
        
        if isinstance(serde,Iterable) and not isinstance(serde,Dict):
            serde_objs.extend(serde)
        else:
            serde_objs.append(serde)

    return serde_objs

config = read_package_support()
objects = get_serde()
