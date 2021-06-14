import torch
from collections import OrderedDict

from .module import repr_to_kwargs

def get_module_args(module: torch.nn.Module) -> OrderedDict:
    """
    get_module_args should extract the relevant args and kwargs to reconstruct the module when deserializing.
    """
    pmodule = type(obj).__module__
    if "transformers" in pmodule:
        return get_transformers_args(module)
    else:
        return get_torch_args(pmodule)


def get_torch_args(module: torch.nn.Module) -> OrderedDict:
    return repr_to_kwargs(module.extra_repr())


def get_transformers_args(module) -> OrderedDict:
    return OrderedDict({"config": module.config})