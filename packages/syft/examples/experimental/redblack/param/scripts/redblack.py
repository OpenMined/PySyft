"""
Redblack v1:

used to scrap python packages to generate .json configuration file
"""

# stdlib
import inspect
import json
import os
from queue import Queue
import sys
import typing
from typing import Any as TypeAny

# third party
from typing_inspect import get_origin

if len(sys.argv) != 2:
    print(f"Usage: {os.path.basename(__file__)} [PACKAGE-NAME]")
    sys.exit(1)
else:
    package_name = sys.argv[1]

try:
    package = __import__(package_name)
except ImportError:
    print(f"Package {package_name} not found...")
    sys.exit(1)


def scraping_lib(root_module: TypeAny) -> TypeAny:
    # root_module = torchvision.transforms.functional

    q = Queue()  # type: ignore
    allowlist = {}
    # print(f'root_module {root_module}')
    q.put(root_module)

    empty_typing_hints = list()
    modules_list = list()
    classes_list = list()
    while not q.empty():

        module = q.get()
        # print(f'Processing {module.__name__}')
        modules_list.append(module.__name__)
        # print(module.__name__)
        # dules_list.append(t.__name__)
        for ax in dir(module):
            # print(ax)
            t = getattr(module, ax)
            if inspect.ismodule(t):
                if module.__name__ in t.__name__:
                    q.put(t)
                # else:
                # print(f'Dont consider {module.__name__}, {ax}, {t.__name__}')
            if inspect.isclass(t):
                # Commenting because allowlist should only contain methods
                # allowlist[module.__name__ + '.' + t.__name__] = module.__name__ + '.' + t.__name__
                classes_list.append(module.__name__ + "." + t.__name__)
            if inspect.ismethod(t) or inspect.isfunction(t):
                # print(f't for debug: {t} {module}')
                try:
                    # try block
                    d = typing.get_type_hints(t)
                    if not d:
                        empty_typing_hints.append(module.__name__ + "." + t.__name__)
                    else:
                        if "return" in d.keys():
                            if isinstance(d["return"], typing._GenericAlias):  # type: ignore
                                # print(type(d['return']))
                                # print(get_origin(d['return']))
                                allowlist[
                                    module.__name__ + "." + t.__name__
                                ] = get_origin(d["return"]).__name__
                            else:
                                # print(d['return'])
                                allowlist[module.__name__ + "." + t.__name__] = d[
                                    "return"
                                ].__name__
                        else:
                            print(f"No return in keys {t}")

                except Exception as e:
                    print(f"get_type_hints didnt work: {e}")

    return allowlist, empty_typing_hints, modules_list, classes_list


allowlist, empty_typing_hints, modules_list, classes_list = scraping_lib(package)

package_support = {}

package_support["lib"] = package_name
package_support["class"] = classes_list
package_support["modules"] = modules_list
package_support["methods"] = allowlist

with open("package_support.json", "w") as outfile:
    json.dump(package_support, outfile)
