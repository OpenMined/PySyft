from .hook import TorchHook
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor
from .tensor import _FixedPrecisionTensor, _TorchTensor, _PlusIsMinusTensor

__all__ = ['TorchHook', '_SyftTensor', '_LocalTensor',
           '_PointerTensor', '_FixedPrecisionTensor', '_TorchTensor', '_PlusIsMinusTensor']

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

# Torch functions we don't want to override
torch.torch_exclude = ['save', 'load', 'typename', 'is_tensor']
