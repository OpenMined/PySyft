#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math

import torch


def implements(torch_function):
    """Register a torch function override for CUDALongTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


HANDLED_FUNCTIONS = {}


class CUDALongTensor(object):
    """
    A wrapper class for `torch.cuda.LongTensor`. When performing operations that are
    currently not supported for `torch.cuda.LongTensor` (e.g `matmul`, `conv2d`), it will
    convert the underlying LongTensor into DoubleTensor and convert the computed
    result back to a LongTensor. The computed result will be the same as the original
    expected result.
    """

    __BITS = torch.iinfo(torch.long).bits
    __N_BLOCKS = 4
    __BLOCK_SIZE = math.ceil(__BITS / __N_BLOCKS)

    __INDICES = []
    __SHIFTS = []
    for i in range(__N_BLOCKS):
        for j in range(__N_BLOCKS):
            if (i + j) * __BLOCK_SIZE >= __BITS:
                continue
            idx = i * __N_BLOCKS + j
            __INDICES.append(idx)
            __SHIFTS.append((i + j) * __BLOCK_SIZE)

    def __init__(self, data=None, device=None):
        r"""
        Construct a CUDALongTensor with `data` on the specified `device`.
        `data` can either be a torch tensor, a CUDALongTensor, or an array-like
        object that can be converted to a torch tensor via torch.as_tensor(data)
        `dtype` of the torch tensor will be automatically converted to torch.long
        regardless of `dtype` of `data`. `device` must be a cuda device.

        Args:
            data (Tensor, array_like, or CUDALongTensor): Initial data for CUDALongTensor.
            device (torch.device): The desired device of CUDALongTensor. Must be a cuda device.
        """

        device = "cuda" if device is None else device
        assert device.startswith("cuda"), "Cannot specify a non-cuda device for CUDALongTensor"

        self._tensor = None
        if data is None:
            return
        if isinstance(data, CUDALongTensor):
            self._tensor = data._tensor
        elif torch.is_tensor(data):
            self._tensor = data.long().to(device)
        else:
            self._tensor = torch.as_tensor(data, dtype=torch.long, device=device)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, CUDALongTensor)) for t in types
        ):
            args = [t.tensor() if hasattr(t, "tensor") else t for t in args]
            result = func(*args, **kwargs)
            if torch.is_tensor(result):
                return CUDALongTensor(result)
            if isinstance(result, list):
                return [CUDALongTensor(t) if torch.is_tensor(t) else t for t in result]
            if isinstance(result, tuple):
                return tuple(CUDALongTensor(t) if torch.is_tensor(t) else t for t in result)
            return result
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self):
        return "CUDALongTensor({})".format(self._tensor)

    def __setitem__(self, index, value):
        self._tensor[index] = value.data

    @property
    def device(self):
        return self._tensor.device

    @property
    def is_cuda(self):
        return self._tensor.is_cuda

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def data(self):
        return self._tensor.data

    @property
    def dtype(self):
        return self._tensor.dtype

    def tensor(self):
        return self._tensor

    def to(self, *args, **kwargs):
        self._tensor = self._tensor.to(*args, **kwargs)
        if not self._tensor.is_cuda:
            return self._tensor
        return self

    def cuda(self, *args, **kwargs):
        self._tensor = self._tensor.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        return self._tensor.cpu(*args, **kwargs)

    def shallow_copy(self):
        """Create a shallow copy of the input tensor."""
        # TODO: Rename this to __copy__()?
        result = CUDALongTensor(self._tensor)
        return result

    def clone(self):
        """Create a deep copy of the input tensor."""
        # TODO: Rename this to __deepcopy__()?
        result = CUDALongTensor()
        result._tensor = self._tensor.clone()
        return result

    @staticmethod
    def __encode_as_fp64(x):
        """Converts a CUDALongTensor `x` to an encoding of
        torch.cuda.DoubleTensor that represent the same data.
        """
        bks = CUDALongTensor.__BLOCK_SIZE
        nb = CUDALongTensor.__N_BLOCKS

        x_block = CUDALongTensor.stack([(x >> (bks * i)) & (2 ** bks - 1) for i in range(nb)])

        return x_block.double()

    @staticmethod
    def __decode_as_int64(x_enc):
        """Converts a CUDALongTensor `x` encoded as torch.cuda.DoubleTensor
        back to the CUDALongTensor it encodes
        """
        x_enc = x_enc.long()

        indices = torch.tensor(CUDALongTensor.__INDICES, device=x_enc.device)
        shifts = torch.tensor(CUDALongTensor.__SHIFTS, device=x_enc.device)
        shifts = shifts.view(-1, *([1] * (x_enc.ndim - 1)))

        x = torch.index_select(x_enc, 0, indices)
        x <<= shifts

        return CUDALongTensor(x.sum(0))

    @staticmethod
    def __patched_conv_ops(op, x, y, *args, **kwargs):
        nb = CUDALongTensor.__N_BLOCKS
        nb2 = nb ** 2

        x_encoded = CUDALongTensor.__encode_as_fp64(x).data
        y_encoded = CUDALongTensor.__encode_as_fp64(y).data

        repeat_idx = [1] * (x_encoded.dim() - 1)
        x_enc_span = x_encoded.repeat(nb, *repeat_idx)
        y_enc_span = torch.repeat_interleave(y_encoded, repeats=nb, dim=0)

        bs, c, *img = x.size()
        c_out, c_in, *ks = y.size()

        x_enc_span = x_enc_span.transpose_(0, 1).reshape(bs, nb2 * c, *img)
        y_enc_span = y_enc_span.reshape(nb2 * c_out, c_in, *ks)

        c_z = c_out if op in ["conv1d", "conv2d"] else c_in

        kwargs["groups"] = nb2
        kwargs["bias"] = None  # TODO

        z_encoded = getattr(torch, f"native_{op}")(x_enc_span, y_enc_span, *args, **kwargs)
        z_encoded = z_encoded.reshape(bs, nb2, c_z, *z_encoded.size()[2:]).transpose_(0, 1)

        r = CUDALongTensor.__decode_as_int64(z_encoded)
        return r

    @staticmethod
    def stack(tensors, *args, **kwargs):
        is_cuda_long = any(hasattr(t, "tensor") for t in tensors)
        tensors = [t.tensor() if hasattr(t, "tensor") else t for t in tensors]
        if is_cuda_long:
            return CUDALongTensor(torch.stack(tensors, *args, **kwargs))
        return torch.stack(tensors, *args, **kwargs)

    @staticmethod
    def cat(tensors, *args, **kwargs):
        is_cuda_long = any(hasattr(t, "tensor") for t in tensors)
        tensors = [t.tensor() if hasattr(t, "tensor") else t for t in tensors]
        if is_cuda_long:
            return CUDALongTensor(torch.cat(tensors, *args, **kwargs))
        return torch.cat(tensors, *args, **kwargs)

    @staticmethod
    @implements(torch.matmul)
    def matmul(x, y, *args, **kwargs):
        nb = CUDALongTensor.__N_BLOCKS

        # Prepend 1 to the dimension of x or y if it is 1-dimensional
        remove_x, remove_y = False, False
        if x.dim() == 1:
            x = x.view(1, x.shape[0])
            remove_x = True
        if y.dim() == 1:
            y = y.view(y.shape[0], 1)
            remove_y = True

        x_encoded = CUDALongTensor.__encode_as_fp64(x).data
        y_encoded = CUDALongTensor.__encode_as_fp64(y).data

        # Span x and y for cross multiplication
        repeat_idx = [1] * (x_encoded.dim() - 1)
        x_enc_span = x_encoded.repeat(nb, *repeat_idx)
        y_enc_span = torch.repeat_interleave(y_encoded, repeats=nb, dim=0)

        # Broadcasting
        for _ in range(abs(x_enc_span.ndim - y_enc_span.ndim)):
            if x_enc_span.ndim > y_enc_span.ndim:
                y_enc_span.unsqueeze_(1)
            else:
                x_enc_span.unsqueeze_(1)

        z_encoded = torch.matmul(x_enc_span, y_enc_span, *args, **kwargs)

        if remove_x:
            z_encoded.squeeze_(-2)
        if remove_y:
            z_encoded.squeeze_(-1)

        return CUDALongTensor.__decode_as_int64(z_encoded)

    @staticmethod
    @implements(torch.conv1d)
    def conv1d(input, weight, *args, **kwargs):
        return CUDALongTensor.__patched_conv_ops("conv1d", input, weight, *args, **kwargs)

    @staticmethod
    @implements(torch.conv_transpose1d)
    def conv_transpose1d(input, weight, *args, **kwargs):
        return CUDALongTensor.__patched_conv_ops("conv_transpose1d", input, weight, *args, **kwargs)

    @staticmethod
    @implements(torch.conv2d)
    def conv2d(input, weight, *args, **kwargs):
        return CUDALongTensor.__patched_conv_ops("conv2d", input, weight, *args, **kwargs)

    @staticmethod
    @implements(torch.conv_transpose2d)
    def conv_transpose2d(input, weight, *args, **kwargs):
        return CUDALongTensor.__patched_conv_ops("conv_transpose2d", input, weight, *args, **kwargs)

    @staticmethod
    @implements(torch.nn.functional.avg_pool2d)
    def avg_pool2d(x, kernel_size, divisor_override=None, *args, **kwargs):
        nb = CUDALongTensor.__N_BLOCKS
        bks = CUDALongTensor.__BLOCK_SIZE

        x_encoded = CUDALongTensor.__encode_as_fp64(x).data

        bs, c, h, w = x.shape
        x_encoded = x_encoded.reshape(nb * bs, c, h, w)

        z_encoded = torch.nn.functional.avg_pool2d(
            x_encoded, kernel_size, divisor_override=1, *args, **kwargs
        )

        z_enc = z_encoded.reshape(nb, bs, *z_encoded.shape[1:]).long()

        z = torch.zeros((nb, bs, *z_encoded.shape[1:]), device=x.device, dtype=torch.long)
        z += z_enc << torch.tensor([bks * i for i in range(nb)], device=x.device).view(
            nb, *([1] * nb)
        )
        z = z.sum(0)

        if isinstance(kernel_size, (int, float)):
            pool_size = kernel_size ** 2
        else:
            pool_size = kernel_size[0] * kernel_size[1]

        if divisor_override is not None:
            z //= divisor_override
        else:
            z //= pool_size

        return CUDALongTensor(z)

    @staticmethod
    @implements(torch.broadcast_tensors)
    def broadcast_tensors(*tensors):
        tensor_list = [t.data for t in tensors]
        results = torch.broadcast_tensors(*tensor_list)
        results = [CUDALongTensor(t) for t in results]
        return results

    def split(self, y, *args, **kwargs):
        splits = self._tensor.split(y, *args, **kwargs)
        splits = [CUDALongTensor(split) for split in splits]
        return splits

    def unbind(self, dim=0):
        results = torch.unbind(self._tensor, dim)
        results = tuple(CUDALongTensor(t) for t in results)
        return results

    def nonzero(self, *args, **kwargs):
        result = self._tensor.nonzero(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(CUDALongTensor(t) for t in result)
        return CUDALongTensor(result)

    def all(self, *args, **kwargs):
        return self._tensor.bool().all(*args, **kwargs)

    def set_(self, source, *args, **kwargs):
        """CUDALongTensor currently does not support inplace set_"""
        self._tensor = source.data
        return self

    def __iadd__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y._tensor
        self._tensor += y
        return self

    def __isub__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor -= y
        return self

    def __imul__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor *= y
        return self

    def __ifloordiv__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor //= y
        return self

    def __idiv__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor /= y
        return self

    def __imod__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor %= y
        return self

    def __iand__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor &= y
        return self

    def __ixor__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor ^= y
        return self

    def __ipow__(self, y):
        if isinstance(y, CUDALongTensor):
            y = y.tensor()
        self._tensor **= y
        return self

    def __and__(self, y):
        result = self.clone()
        return result.__iand__(y)

    def __xor__(self, y):
        result = self.clone()
        return result.__ixor__(y)

    def __add__(self, y):
        result = self.clone()
        return result.__iadd__(y)

    def __sub__(self, y):
        result = self.clone()
        return result.__isub__(y)

    def __rsub__(self, y):
        result = self.clone()
        result._tensor = y - result._tensor
        return result

    def __mul__(self, y):
        result = self.clone()
        return result.__imul__(y)

    def __floordiv__(self, y):
        result = self.clone()
        return result.__ifloordiv__(y)

    def __truediv__(self, y):
        result = self.clone()
        return result.__idiv__(y)

    def __mod__(self, y):
        result = self.clone()
        return result.__imod__(y)

    def __pow__(self, y):
        result = self.clone()
        return result.__ipow__(y)

    def __neg__(self):
        result = self.clone()
        result._tensor = -result._tensor
        return result

    def __eq__(self, y):
        return CUDALongTensor(self._tensor == y)

    def __ne__(self, y):
        return CUDALongTensor(self._tensor != y)

    def __lt__(self, y):
        return CUDALongTensor(self._tensor < y)

    def __gt__(self, y):
        return CUDALongTensor(self._tensor > y)

    def __le__(self, y):
        return CUDALongTensor(self._tensor <= y)

    def __ge__(self, y):
        return CUDALongTensor(self._tensor >= y)

    def lshift_(self, value):
        """Right shift elements by `value` bits"""
        assert isinstance(value, int), "lshift must take an integer argument."
        self._tensor <<= value
        return self

    def lshift(self, value):
        """Left shift elements by `value` bits"""
        return self.clone().lshift_(value)

    def rshift_(self, value):
        """Right shift elements by `value` bits"""
        assert isinstance(value, int), "rshift must take an integer argument."
        self._tensor >>= value
        return self

    def rshift(self, value):
        """Right shift elements by `value` bits"""
        return self.clone().rshift_(value)

    __lshift__ = lshift
    __rshift__ = rshift

    # In-place bitwise operators
    __ilshift__ = lshift_
    __irshift__ = rshift_

    __radd__ = __add__
    __rmul__ = __mul__
    __rpow__ = __pow__


REGULAR_FUNCTIONS = [
    "__getitem__",
    "index_select",
    "view",
    "flatten",
    "t",
    "transpose",
    "unsqueeze",
    "repeat",
    "squeeze",
    "narrow",
    "expand",
    "roll",
    "unfold",
    "flip",
    "trace",
    "prod",
    "sum",
    "cumsum",
    "reshape",
    "permute",
    "pow",
    "float",
    "long",
    "double",
    "scatter",
    "scatter_add",
    "index_fill",
    "index_add",
    "take",
    "gather",
    "where",
    "add",
    "sub",
    "mul",
    "div",
    "le",
    "ge",
    "gt",
    "lt",
    "eq",
    "ne",
    "neg",
    "abs",
    "sign",
]

PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel", "item"]

INPLACE_FUNCTIONS = [
    "add_",
    "sub_",
    "mul_",
    "div_",
    "copy_",
    "abs_",
    "neg_",
    "index_fill_",
    "index_add_",
    "scatter_",
    "scatter_add_",
    "le_",
    "ge_",
    "gt_",
    "lt_",
    "eq_",
    "ne_",
    "neg_",
    "abs_",
    "sign_",
]


def _add_regular_function(func_name):
    """
    Adds function to `CUDALongTensor` that is applied directly on the underlying
    `_tensor` attribute, and stores the result in the same attribute.
    """

    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        args = [t.tensor() if hasattr(t, "tensor") else t for t in args]
        for key, value in kwargs.items():
            if hasattr(value, "tensor"):
                kwargs[key] = value.tensor()
        result._tensor = getattr(result._tensor, func_name)(*args, **kwargs)
        return result

    setattr(CUDALongTensor, func_name, regular_func)


def _add_property_function(func_name):
    """
    Adds function to `CUDALongTensor` that is applied directly on the underlying
    `_tensor` attribute, and returns the result of that function.
    """

    def property_func(self, *args, **kwargs):
        result = getattr(self._tensor, func_name)(*args, **kwargs)
        return result

    setattr(CUDALongTensor, func_name, property_func)


def _add_inplace_function(func_name):
    """
    Adds function to `CUDALongTensor` that is applied in place on the underlying
    `_tensor` attribute, and returns the result of that function.
    """

    def inplace_func(self, *args, **kwargs):
        args = [t.tensor() if hasattr(t, "tensor") else t for t in args]
        for key, value in kwargs.items():
            if hasattr(value, "tensor"):
                kwargs[key] = value.tensor()

        result = getattr(self._tensor, func_name)(*args, **kwargs)
        self._tensor.set_(result)
        return self

    setattr(CUDALongTensor, func_name, inplace_func)


for func_name in REGULAR_FUNCTIONS:
    _add_regular_function(func_name)

for func_name in PROPERTY_FUNCTIONS:
    _add_property_function(func_name)

for func_name in INPLACE_FUNCTIONS:
    _add_inplace_function(func_name)
