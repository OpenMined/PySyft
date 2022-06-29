# third party
import numpy as np

# relative
from ....common.serde.serializable import serializable
from ...autodp.phi_tensor import PhiTensor
from ..utils import col2im_indices, im2col_indices
from .base import Layer


@serializable(recursive_serde=True)
class AvgPool(Layer):
    """Average pooling operation for spatial data.
    Parameters
    ----------
    pool_size : tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.
    Returns
    -------
    4D numpy.array
        with shape `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """

    __attr_allowlist__ = ("pool_size", "input_shape", "out_shape", "last_input")

    def __init__(self, pool_size, stride=1):
        if isinstance(pool_size, tuple):
            self.pool_size = pool_size
        elif isinstance(pool_size, int):
            self.pool_size = pool_size, pool_size
        else:
            raise TypeError("Pool size can be either int or Tuple")

        self.stride = stride

        self.input_shape = None
        self.out_shape = None
        self.last_input = None

    def connect_to(self, prev_layer):
        assert 5 > len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        
        new_h = (old_h - pool_h) // self.stride + 1
        new_w = (old_w - pool_w) // self.stride + 1

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

    def forward(self, input: PhiTensor, *args, **kwargs):
        # shape

        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        h_out, w_out = self.out_shape[-2:]

        # forward
        self.last_input = input
        
        n, d, h, w = input.shape
        input_reshaped = input.reshape(n * d, 1, h, w)
        self.X_col = im2col_indices(input_reshaped, pool_h, pool_w, padding=0, stride=self.stride)

        outputs = self.X_col.mean(axis=0)

        outputs = outputs.reshape(h_out, w_out, n, d)
        outputs = outputs.transpose(2, 3, 0, 1)

        return outputs

    def backward(self, pre_grad: PhiTensor, *args, **kwargs):

        n, d, w, h = self.input_shape
        pool_h, pool_w = self.pool_size

        dX_col = self.X_col.zeros_like()

        dout_col = pre_grad.transpose(2, 3, 0, 1).ravel()

        dout_col_size = np.prod(dout_col.shape)

        dX_col[:, range(dout_col_size)] = dout_col * (1.0 / dX_col.shape[0])

        dX = col2im_indices(dX_col, (n * d, 1, h, w), pool_h, pool_w, padding=0, stride=self.stride)

        dX = dX.reshape(self.input_shape)

        return dX


@serializable(recursive_serde=True)
class MaxPool(Layer):
    """Max pooling operation for spatial data.
    Parameters
    ----------
    pool_size : tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.
    Returns
    -------
    4D numpy.array
        with shape `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """

    __attr_allowlist__ = ("pool_size", "input_shape", "out_shape", "last_input")

    def __init__(self, pool_size, stride=1):
        if isinstance(pool_size, tuple):
            self.pool_size = pool_size
        elif isinstance(pool_size, int):
            self.pool_size = pool_size, pool_size
        else:
            raise TypeError("Pool size can be either int or Tuple")

        self.stride = stride

        self.input_shape = None
        self.out_shape = None
        self.last_input = None
        self.X_col = None
        self.max_idx = None

    def connect_to(self, prev_layer):
        # prev_layer.out_shape: (nb_batch, ..., height, width)
        assert len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h = (old_h - pool_h) // self.stride + 1
        new_w = (old_w - pool_w) // self.stride + 1

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

    def forward(self, input: PhiTensor, *args, **kwargs):
        # shape

        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        h_out, w_out = self.out_shape[-2:]

        # forward
        self.last_input = input

        n, d, h, w = input.shape
        input_reshaped = input.reshape((n * d, 1, h, w))
        self.X_col = im2col_indices(input_reshaped, pool_h, pool_w, padding=0, stride=self.stride)

        self.max_idx = self.X_col._argmax(axis=0)

        return self.X_col, self.max_idx, h_out, w_out, n, d

        outputs = self.X_col[self.max_idx, range(self.max_idx.size)]

        outputs = outputs.reshape((h_out, w_out, n, d))
        outputs = outputs.transpose((2, 3, 0, 1))

        return outputs

    def backward(self, pre_grad: PhiTensor, *args, **kwargs):

        n, d, w, h = self.input_shape
        pool_h, pool_w = self.pool_size

        dX_col = self.X_col.zeros_like()

        dout_col = pre_grad.transpose((2, 3, 0, 1)).ravel()

        dout_col_size = np.prod(dout_col.shape)

        dX_col[self.max_idx, range(dout_col_size)] = dout_col

        dX = col2im_indices(dX_col, (n * d, 1, h, w), pool_h, pool_w, padding=0, stride=self.stride)

        dX = dX.reshape(self.input_shape)

        return dX
