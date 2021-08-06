# import syft as sy
import sys
import importlib
import ast
import wrapt

# hook_loaded = set()
# sys.modules["xgboost"]
modules_imported = set()
old_import = __import__

id(__builtins__.__import__)

queue=[]

def my_import(module, *args, **kwargs):
    if module not in modules_imported and not module.startswith("syft") and not module.startswith("_") and "." not in module:
    # if not module.startswith("syft") and "." not in module:
        queue.append(f"syft_{module}")
        modules_imported.add(module)
    return old_import(module, *args, **kwargs)


@wrapt.when_imported("xgboost")
def post_import_hook(module):
    print("hook called")
    print(queue)
    while queue:
        syft_module=queue[0]
        try:
            importlib.import_module(syft_module)
            globals()[syft_module]=sys.modules[syft_module]
        except Exception as e:
            print(f"Failed to load {syft_module}\n {e}")
        queue.pop(0)

__builtins__.__import__ = my_import

id(__builtins__.__import__)

# def my_import(module, *args, **kwargs):
#     modules_imported.add(module)
#     #     print(module, 'loaded successfully')
#     if module == "xgboost":
#         return old_import(module, *args, **kwargs)
#     return old_import(module, *args, **kwargs)

import xgboost

print(xgboost)
print(syft_xgboost)
print(sys.path)