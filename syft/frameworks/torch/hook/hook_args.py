import torch

from syft.frameworks.torch.tensors.interpreters.native import TorchTensor
from syft.generic.frameworks.hook.hook_args import register_backward_func
from syft.generic.frameworks.hook.hook_args import register_forward_func
from syft.generic.frameworks.hook.hook_args import register_type_rule
from syft.generic.frameworks.hook.hook_args import one

from syft.exceptions import PureFrameworkTensorFoundError


type_rule = {torch.Tensor: one, torch.nn.Parameter: one}

forward_func = {
    torch.Tensor: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    torch.nn.Parameter: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
}

backward_func = {
    TorchTensor: lambda i: i.wrap(),
    torch.Tensor: lambda i: i.wrap(),
    torch.nn.Parameter: lambda i: torch.nn.Parameter(data=i),
}

register_type_rule(type_rule)
register_forward_func(forward_func)
register_backward_func(backward_func)
