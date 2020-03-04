from functools import wraps

import crypten
from syft.generic.frameworks.hook.trace import tracer
from syft.frameworks.torch.tensors.crypten.syft_crypten import SyftCrypTensor
import torch as th


def get_hooked_crypten_func(func_api_name, func):
    cmd_name = f"crypten.{func_api_name}"

    @tracer(func_name=cmd_name)
    @wraps(func)
    def overloaded_func(*args, **kwargs):
        try:
            response = SyftCrypTensor(tensor=func(*args, **kwargs)).wrap()
        except RuntimeError:
            response = SyftCrypTensor(tensor=th.zeros([])).wrap()

        return response

    return overloaded_func
