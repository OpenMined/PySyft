# stdlib
import sys
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

scopes_cache = {}


def generate_proxy_func(module: ModuleType, method_name: str) -> Callable:
    def proxy_func(*args: Any, **kwargs: Any) -> Any:
        object = getattr(module, method_name)

        if not callable(object):
            return object

        return object(*args, **kwargs)

    return proxy_func


class Scope(type):
    def __new__(cls, name: str, bases: tuple, dct: Dict[str, Any]) -> Any:
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def from_qualname(module_path: str) -> Tuple["Scope", str]:
        module = sys.modules[module_path]
        path = module_path.split(".")
        functions: Dict[str, Any] = {}

        for method_name in dir(module):
            functions[method_name] = generate_proxy_func(module, method_name)

        name = "".join(path) + "_global_scope"
        functions["__qualname__"] = "syft.lib.misc.scope." + name
        functions["__name__"] = name

        type = Scope(name, tuple(), functions)

        globals()[name] = type
        scopes_cache[name] = type

        return type, type.__qualname__
