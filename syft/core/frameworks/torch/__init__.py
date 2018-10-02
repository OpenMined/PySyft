from .hook import TorchHook
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor
from .tensor import _FixedPrecisionTensor, _TorchTensor, _PlusIsMinusTensor, _GeneralizedPointerTensor
from .tensor import _SPDZTensor, _SNNTensor

__all__ = ['TorchHook', '_SyftTensor', '_LocalTensor',
           '_PointerTensor', '_FixedPrecisionTensor', '_TorchTensor',
           '_PlusIsMinusTensor', '_GeneralizedPointerTensor', '_SPDZTensor',
           '_SNNTensor']

import torch

# this is a list of all module functions in the torch module
torch.torch_funcs = dir(torch)

# this is a list of all module functions in torch.nn.functional
torch.torch_functional_funcs = dir(torch.nn.functional)

# Gathers all the functions from above
torch.torch_modules = {
    'torch': torch.torch_funcs,
    'torch.nn.functional': torch.torch_functional_funcs
}

# this is the list of torch tensor types that we will override for remote execution
torch.tensor_types = [torch.FloatTensor,
                      torch.DoubleTensor,
                      torch.HalfTensor,
                      torch.ByteTensor,
                      torch.CharTensor,
                      torch.ShortTensor,
                      torch.IntTensor,
                      torch.LongTensor]

torch.var_types = [torch.autograd.variable.Variable, torch.nn.Parameter]

# a list of all classes in which we will override their methods for remote execution
torch.tensorvar_types = torch.tensor_types + \
                        [torch.autograd.variable.Variable]

torch.tensorvar_types_strs = [x.__name__ for x in torch.tensorvar_types]

torch.tensorvar_methods = list(
    set(
        [method
         for tensorvar in torch.tensorvar_types
         for method in dir(tensorvar)]
    )
)
torch.tensorvar_methods.append('get_shape')
torch.tensorvar_methods.append("share")
torch.tensorvar_methods.append("fix_precision")
torch.tensorvar_methods.append("decode")

# Torch functions we don't want to override
torch.torch_exclude = ['save', 'load', 'typename', 'is_tensor', 'manual_seed']

torch.guard = {
    'syft.core.frameworks.torch.tensor.Variable': torch.autograd.Variable,
    'syft.core.frameworks.torch.tensor._PointerTensor': _PointerTensor,
    'syft.core.frameworks.torch.tensor._SyftTensor': _SyftTensor,
    'syft.core.frameworks.torch.tensor._LocalTensor': _LocalTensor,
    'syft.core.frameworks.torch.tensor._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.core.frameworks.torch.tensor._GeneralizedPointerTensor': _GeneralizedPointerTensor,
    'syft._PlusIsMinusTensor': _PlusIsMinusTensor,
    'syft._SPDZTensor': _SPDZTensor,
    'syft._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.core.frameworks.torch.tensor.FloatTensor': torch.FloatTensor,
    'syft.core.frameworks.torch.tensor.DoubleTensor': torch.DoubleTensor,
    'syft.core.frameworks.torch.tensor.HalfTensor': torch.HalfTensor,
    'syft.core.frameworks.torch.tensor.ByteTensor': torch.ByteTensor,
    'syft.core.frameworks.torch.tensor.CharTensor': torch.CharTensor,
    'syft.core.frameworks.torch.tensor.ShortTensor': torch.ShortTensor,
    'syft.core.frameworks.torch.tensor.IntTensor': torch.IntTensor,
    'syft.core.frameworks.torch.tensor.LongTensor': torch.LongTensor,
    'syft.Variable': torch.autograd.Variable,
    'syft.FloatTensor': torch.FloatTensor,
    'syft.DoubleTensor': torch.DoubleTensor,
    'syft.HalfTensor': torch.HalfTensor,
    'syft.ByteTensor': torch.ByteTensor,
    'syft.CharTensor': torch.CharTensor,
    'syft.ShortTensor': torch.ShortTensor,
    'syft.IntTensor': torch.IntTensor,
    'syft.LongTensor': torch.LongTensor,
    'syft.Parameter': torch.nn.Parameter
}


def _command_guard(command, allowed):
    if isinstance(allowed, dict):
        allowed_names = []
        for module_name, func_names in allowed.items():
            for func_name in func_names:
                allowed_names.append(module_name + '.' + func_name)
        allowed = allowed_names
    if command not in allowed:
        raise RuntimeError(
            'Command "{}" is not a supported Torch operation.'.format(command))
    return command

torch._command_guard = _command_guard


def _is_command_valid_guard(command, allowed):
    try:
        torch._command_guard(command, allowed)
    except RuntimeError:
        return False
    return True

torch._is_command_valid_guard = _is_command_valid_guard
