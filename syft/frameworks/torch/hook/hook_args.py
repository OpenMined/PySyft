import torch

from syft.frameworks.torch.tensors.interpreters.native import TorchTensor
from syft.generic.frameworks.hook.hook_args import (
    register_ambiguous_method,
    register_ambiguous_function,
    register_backward_func,
    register_forward_func,
    register_type_rule,
    one,
)

from syft.exceptions import PureFrameworkTensorFoundError

type_rule = {
    torch.Tensor: one,
    torch.nn.Parameter: one,
}

forward_func = {
    torch.Tensor: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    torch.nn.Parameter: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
}

backward_func = {
    TorchTensor: lambda i, **kwargs: i.wrap(**kwargs),
    torch.Tensor: lambda i, **kwargs: i.wrap(**kwargs),
    torch.nn.Parameter: lambda i, **kwargs: torch.nn.Parameter(data=i),
}

# Methods or functions whose signature changes a lot and that we don't want to "cache", because
# they have an arbitrary number of tensors in args which can trigger unexpected behaviour
ambiguous_methods = {
    "__getitem__",
    "__setitem__",
    "_getitem_public",
    "add_",
    "backward",
    "cat",
    "chunk",
    "new",
    "permute",
    "reshape",
    "split",
    "stack",
    "sub_",
    "view",
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
    "torch.split",
    "split",
    "backward",
}

register_ambiguous_method(*ambiguous_methods)
register_ambiguous_function(*ambiguous_functions)
register_type_rule(type_rule)
register_forward_func(forward_func)
register_backward_func(backward_func)
