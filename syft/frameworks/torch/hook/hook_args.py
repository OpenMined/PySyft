import torch

from syft import dependency_check
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
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

type_rule = {
    torch.Tensor: one,
    torch.nn.Parameter: one,
    AutogradTensor: one,
    LoggingTensor: one,
    PaillierTensor: one,
}

forward_func = {
    torch.Tensor: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    torch.nn.Parameter: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    AutogradTensor: get_child,
    LoggingTensor: get_child,
    PaillierTensor: get_child,
}

backward_func = {
    TorchTensor: lambda i, **kwargs: i.wrap(**kwargs),
    torch.Tensor: lambda i, **kwargs: i.wrap(**kwargs),
    torch.nn.Parameter: lambda i, **kwargs: torch.nn.Parameter(data=i),
    AutogradTensor: lambda i, **kwargs: AutogradTensor(data=i).on(i, wrap=False),
    LoggingTensor: lambda i, **kwargs: LoggingTensor().on(i, wrap=False),
    PaillierTensor: lambda i, **kwargs: PaillierTensor().on(i, wrap=False),
}

if dependency_check.crypten_available:
    import crypten

    type_rule[crypten.mpc.MPCTensor] = one
    forward_func[crypten.mpc.MPCTensor] = (
        lambda i: i.child if hasattr(i, "child") else ().throw(PureFrameworkTensorFoundError)
    )
    backward_func[crypten.mpc.MPCTensor] = lambda i, **kwargs: i.wrap(**kwargs)


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
