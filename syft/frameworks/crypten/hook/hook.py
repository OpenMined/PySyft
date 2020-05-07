from functools import wraps

import syft

import crypten
import torch as th

methods_to_hook = ["load_from_party"]


def hook_plan_building():
    """When builing the plan we should not call directly specific
    methods from CrypTen and as such we return here some "dummy" responses
    only to build the plan.
    """

    f = lambda *args, **kwargs: crypten.cryptensor(th.zeros([]))
    for method_name in methods_to_hook:
        method = getattr(crypten, method_name)
        setattr(crypten, f"buildplan_{method_name}", method)
        setattr(crypten, method_name, f)


def unhook_plan_building():
    """After building the plan we unhook the methods such that
    we call the "real" methods in the actual workers
    """
    for method_name in methods_to_hook:
        method = getattr(crypten, f"buildplan_{method_name}")
        setattr(crypten, method_name, method)
