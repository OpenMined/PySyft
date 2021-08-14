# import syft as sy
# stdlib
import ast
import importlib
import sys

# third party
import wrapt

# hook_loaded = set()
# sys.modules["xgboost"]
modules_imported = set()
old_import = __import__

# id(__builtins__.__import__)

queue = []


def syft_lib_import(module, *args, **kwargs):
    if (
        module not in modules_imported
        and not module.startswith("syft")
        and not module.startswith("_")
        and "." not in module
    ):
        # if not module.startswith("syft") and "." not in module:
        queue.append(f"syft_{module}")
        modules_imported.add(module)
    return old_import(module, *args, **kwargs)


@wrapt.when_imported("xgboost")
def post_import_hook(module):
    with open("sy_libs_log.txt", "w+") as f:
        f.write("hook called\n")
        f.write(f"{str(queue)}\n")
        while queue:
            syft_module = queue[0]
            try:
                importlib.import_module(syft_module)
                globals()[syft_module] = sys.modules[syft_module]
            except Exception as e:
                f.write(f"Failed to load {syft_module}\n {e}\n")
            queue.pop(0)


__builtins__.__import__ = syft_lib_import

# id(__builtins__.__import__)


# third party
import xgboost

print(xgboost)
print(syft_xgboost)
# print(sys.path)
