from functools import wraps

import syft

import crypten
from syft.generic.frameworks.hook.trace import tracer
import torch as th


def get_hooked_crypten_func(func_api_name, func):
    cmd_name = f"crypten.{func_api_name}"

    @tracer(func_name=cmd_name)
    @wraps(func)
    def overloaded_func(*args, **kwargs):
        if syft.hook.trace.active:
            response = crypten.cryptensor(th.zeros([]))
        else:
            response = func(*args, **kwargs)

        return response

    return overloaded_func
