from .hook import TorchHook
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor
from .tensor import _FixedPrecisionTensor, _TorchTensor

__all__ = ['TorchHook', '_SyftTensor', '_LocalTensor',
           '_PointerTensor', '_FixedPrecisionTensor', '_TorchTensor']

import torch

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