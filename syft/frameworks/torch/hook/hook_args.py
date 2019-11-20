import torch

from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.native import TorchTensor
from syft.generic.frameworks.hook.hook_args import (
    get_child,
    register_ambiguous_method,
    register_ambiguous_function,
    register_backward_func,
    register_forward_func,
    register_type_rule,
    one,
)

from syft.exceptions import PureFrameworkTensorFoundError


type_rule = {torch.Tensor: one, torch.nn.Parameter: one, AutogradTensor: one, LoggingTensor: one}

forward_func = {
    torch.Tensor: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    torch.nn.Parameter: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    AutogradTensor: get_child,
    LoggingTensor: get_child,
}

backward_func = {
    TorchTensor: lambda i: i.wrap(),
    torch.Tensor: lambda i: i.wrap(),
    torch.nn.Parameter: lambda i: torch.nn.Parameter(data=i),
    AutogradTensor: lambda i: AutogradTensor(data=i).on(i, wrap=False),
    LoggingTensor: lambda i: LoggingTensor().on(i, wrap=False),
}

ambiguous_methods = {
    "__getitem__",
    "_getitem_public",
    "__setitem__",
    "view",
    "permute",
    "add_",
    "sub_",
    "new",
    "chunk",
}

ambiguous_functions = {
    "torch.unbind",
    "unbind",
    "torch.stack",
    "stack",
    "torch.cat",
    "cat",
    "torch.mean",
    "torch.sum",
    "torch.chunk",
    "chunk",
    "torch.functional.split",
    "split",
    "backward",
}

register_ambiguous_method(*ambiguous_methods)
register_ambiguous_function(*ambiguous_functions)
register_type_rule(type_rule)
register_forward_func(forward_func)
register_backward_func(backward_func)
