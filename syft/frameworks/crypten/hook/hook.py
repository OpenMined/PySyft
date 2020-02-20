from functools import wraps

import crypten
from syft.generic.frameworks.hook.trace import tracer
import torch as th


def get_hooked_crypten_func(func_api_name, func):
    cmd_name = f"crypten.{func_api_name}"

    @tracer(func_name=cmd_name)
    @wraps(func)
    def overloaded_func(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
        except Exception as e:
            response = th.tensor([1,2])

        return response

    return overloaded_func
