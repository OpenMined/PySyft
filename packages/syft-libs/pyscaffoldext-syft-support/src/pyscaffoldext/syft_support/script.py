from importlib import import_module
from queue import Queue
import inspect 
import typing
import json
def scraping_lib(thing): 
    # thing = torchvision.transforms.functional

    q = Queue()
    allowlist = {}
    # print(f'thing {thing}')
    q.put(thing)

    empty_typing_hints = list()
    modules_list = list()
    classes_list = list()
    while(not q.empty()):

        module = q.get()
        # print(f'Processing {module.__name__}')
        modules_list.append(module.__name__)
        # print(module.__name__)
        # dules_list.append(t.__name__)
        for ax in dir(module):
            # print(ax)
            t = getattr(module, ax)
            
            if (inspect.ismodule(t)):
                if module.__name__ in t.__name__:
                    q.put(t)
                # else:
                    # print(f'Dont consider {module.__name__}, {ax}, {t.__name__}')
            if (inspect.isclass(t)):
                # Commenting because allowlist should only contain methods
                # allowlist[module.__name__ + '.' + t.__name__] = module.__name__ + '.' + t.__name__
                classes_list.append(module.__name__ + '.' + t.__name__)
                #q.put(t)
            if (inspect.ismethod(t) or inspect.isfunction(t)):
                # print(f't for debug: {t} {module}')
                try:
                    # try block
                    d = typing.get_type_hints(t)
                    if (not d): 
                        empty_typing_hints.append(module.__name__ + '.' + t.__name__)
                    else:
                        if 'return' in d.keys():
                            if isinstance(d['return'], typing._GenericAlias):
                                # print(type(d['return']))
                                # print(get_origin(d['return']))
                                allowlist[module.__name__ + '.' + t.__name__] = get_origin(d['return']).__name__
                            else:
                                # print(d['return'])
                                allowlist[module.__name__ + '.' + t.__name__] = d['return'].__name__
                        else:
                            print(f'No return in keys {t}')

                except Exception as e:
                    print(f'get_type_hints didnt work: {e}')
                
    
    return allowlist, empty_typing_hints, modules_list, classes_list

def generate_package_support(lib):
    allowlist,empty_typing_hints, modules_list, classes_list = scraping_lib(lib)
    package_support = {}

    package_support['lib'] = lib.__name__
    package_support['class'] = classes_list
    package_support['modules'] = modules_list
    package_support['methods'] = allowlist

    return json.dumps(package_support)