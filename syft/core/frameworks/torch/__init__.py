from .hook import TorchHook
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor
from .tensor import _FixedPrecisionTensor, _TorchTensor, _PlusIsMinusTensor, _GeneralizedPointerTensor
from .tensor import _SPDZTensor, _SNNTensor
from enum import Enum, auto

__all__ = ['TorchHook', '_SyftTensor', '_LocalTensor',
           '_PointerTensor', '_FixedPrecisionTensor', '_TorchTensor',
           '_PlusIsMinusTensor', '_GeneralizedPointerTensor', '_SPDZTensor',
           '_SNNTensor']

import torch

torch.encode_timer = 0
torch.handle_call_timer = 0
torch.execute_call_timer = 0

# this is a list of all module functions in the torch module
torch.torch_funcs = dir(torch)

# this is a list of all module functions in torch.nn.functional
torch.torch_functional_funcs = dir(torch.nn.functional)

# Gathers all the functions from above
torch.torch_modules = {
    'torch': torch.torch_funcs,
    'torch.nn.functional': torch.torch_functional_funcs
}
# 'torch.nn.functional': torch.torch_functional_funcs

# this is the list of torch tensor types that we will override for remote execution
torch.tensor_types = [torch.FloatTensor,
                      torch.DoubleTensor,
                      torch.HalfTensor,
                      torch.ByteTensor,
                      torch.CharTensor,
                      torch.ShortTensor,
                      torch.IntTensor,
                      torch.LongTensor]
torch.tensor_types_tuple = tuple(torch.tensor_types)

torch.var_types = [torch.autograd.variable.Variable, torch.nn.Parameter]
torch.var_types_tuple = tuple(torch.var_types)

# a list of all classes in which we will override their methods for remote execution
torch.tensorvar_types = torch.tensor_types + [torch.autograd.variable.Variable]

torch.tensorvar_types_strs = [x.__name__ for x in torch.tensorvar_types]

torch.syft_tensor_name = None
torch.tensor_type_names = [x.__name__ for x in torch.tensor_types]
torch.var_type_names = [x.__name__ for x in torch.var_types] + ['syft.Variable', 'syft.Parameter']

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
    '_PlusIsMinusTensor': _PlusIsMinusTensor,
    '_SPDZTensor': _SPDZTensor,
    '_FixedPrecisionTensor': _FixedPrecisionTensor,
    '_SNNTensor': _SNNTensor,
    'Variable': torch.autograd.Variable,
    'FloatTensor': torch.FloatTensor,
    'DoubleTensor': torch.DoubleTensor,
    'HalfTensor': torch.HalfTensor,
    'ByteTensor': torch.ByteTensor,
    'CharTensor': torch.CharTensor,
    'ShortTensor': torch.ShortTensor,
    'IntTensor': torch.IntTensor,
    'LongTensor': torch.LongTensor,
    'Parameter': torch.nn.Parameter
}
keys = list(torch.guard.keys())
for key in keys:
    torch.guard['syft.' + key] = torch.guard[key]


def get_allowed_command(allowed):
    if isinstance(allowed, dict):
        allowed_names = set()
        for module_name, func_names in allowed.items():
            for func_name in func_names:
                allowed_names.add(module_name + '.' + func_name)
        allowed = allowed_names
    return allowed

allowed_commands = {
    'tensorvar_methods': get_allowed_command(torch.tensorvar_methods),
    'torch_modules': get_allowed_command(torch.torch_modules)
}


def get_native_torch_name(attr):
    elems = attr.split('.')
    elems[-1] = 'native_' + elems[-1]
    native_func_name = '.'.join(elems)
    return native_func_name

native_commands = {
    'tensorvar_methods': {
        cmd: 'native_' + cmd for cmd in allowed_commands['tensorvar_methods']
    },
    'torch_modules': {
        cmd: get_native_torch_name(cmd) for cmd in allowed_commands['torch_modules']
    }
}


def eval_torch_modules():
    for cmd_name, native_cmd_name in native_commands['torch_modules'].items():
        if cmd_name not in torch.torch_exclude:
            try:
                native_commands['torch_modules'][cmd_name] = eval(native_cmd_name)
            except AttributeError:
                native_commands['torch_modules'][cmd_name] = eval(cmd_name)
        else:
            native_commands['torch_modules'][cmd_name] = eval(cmd_name)

torch.eval_torch_modules = eval_torch_modules



def _command_guard(command, torch_domain, get_native=False):
    if command not in allowed_commands[torch_domain]:
        raise RuntimeError(
            'Command "{}" is not a supported Torch operation.'.format(command))
    if get_native:
        return native_commands[torch_domain][command]
    return command

torch._command_guard = _command_guard


def _is_command_valid_guard(command, allowed):
    try:
        torch._command_guard(command, allowed)
    except RuntimeError:
        return False
    return True

torch._is_command_valid_guard = _is_command_valid_guard
