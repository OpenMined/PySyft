# stdlib
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import NDArray

# relative
from ....common.serde.serializable import serializable
from ...autodp.gamma_tensor import GammaTensor
from ...autodp.phi_tensor import PhiTensor
from ..utils import col2im_indices
from ..utils import im2col_indices
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

    __attr_allowlist__ = (
        "stride",
        "pool_size",
        "input_shape",
        "out_shape",
        "X_col",
    )

    def __init__(self, pool_size: Union[Tuple[int, ...], int], stride: int = 1) -> None:
        if isinstance(pool_size, tuple):
            self.pool_size = pool_size
        elif isinstance(pool_size, int):
            self.pool_size = pool_size, pool_size
        else:
            raise TypeError("Pool size can be either int or Tuple")

        self.stride = stride

        self.input_shape: Optional[Tuple[int, ...]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None

    def connect_to(self, prev_layer: Layer) -> None:

        if not (5 > len(prev_layer.out_shape) >= 3):  # type: ignore
            raise ValueError(
                f"Input layer shape:{prev_layer.out_shape} dimension should be between [3,5)"  # type: ignore
            )

        old_h, old_w = prev_layer.out_shape[-2:]  # type: ignore
        pool_h, pool_w = self.pool_size

        new_h = (old_h - pool_h) // self.stride + 1
        new_w = (old_w - pool_w) // self.stride + 1

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)  # type: ignore

    def forward(
        self, input: Union[PhiTensor, GammaTensor], *args: Any, **kwargs: Any
    ) -> Union[GammaTensor, PhiTensor]:
        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size

        if self.out_shape is None:
            raise ValueError(
                "`self.out_shape` is None. Please check the out_shape is correctly initialized."
            )

        h_out, w_out = self.out_shape[-2:]

        # forward

        n, d, h, w = input.shape
        input_reshaped = input.reshape((n * d, 1, h, w))

        self.X_col = im2col_indices(
            input_reshaped, pool_h, pool_w, padding=0, stride=self.stride
        )
        outputs = self.X_col.mean(axis=0)
        outputs = outputs.reshape((h_out, w_out, n, d))
        outputs = outputs.transpose((2, 3, 0, 1))
        # print("Done with AvgPool forward pass")
        return outputs

    def backward(
        self, pre_grad: Union[PhiTensor, GammaTensor], *args: Any, **kwargs: Any
    ) -> Union[GammaTensor, PhiTensor]:

        if self.input_shape is None:
            raise ValueError(
                "Input shape is None. \
                Please check if the input shape is correctly initialized from the prev_layer."
            )

        n, d, w, h = self.input_shape
        pool_h, pool_w = self.pool_size

        dX_col = self.X_col.zeros_like()

        dout_col = pre_grad.transpose(2, 3, 0, 1).ravel()

        dout_col_size = np.prod(dout_col.shape)

        dX_col[:, range(dout_col_size)] = dout_col * (1.0 / dX_col.shape[0])  # type: ignore

        dX = col2im_indices(
            dX_col, (n * d, 1, h, w), pool_h, pool_w, padding=0, stride=self.stride
        )

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

    __attr_allowlist__ = (
        "stride",
        "X_col",
        "max_idx",
        "pool_size",
        "input_shape",
        "out_shape",
    )

    def __init__(self, pool_size: Union[Tuple[int, ...], int], stride: int = 1) -> None:
        if isinstance(pool_size, tuple):
            self.pool_size = pool_size
        elif isinstance(pool_size, int):
            self.pool_size = pool_size, pool_size
        else:
            raise TypeError("Pool size can be either int or Tuple")

        self.stride = stride

        self.input_shape: Optional[Tuple[int, ...]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None
        self.X_col: Optional[Union[GammaTensor, PhiTensor]] = None
        self.max_idx: Optional[Union[int, slice, range, NDArray]] = None

    def connect_to(self, prev_layer: Layer) -> None:
        # prev_layer.out_shape: (nb_batch, ..., height, width)
        if len(prev_layer.out_shape) < 3:  # type: ignore
            raise ValueError(
                f"Input layer shape should have at least \
                three dimensions:{prev_layer.out_shape} for MaxPool"  # type: ignore
            )

        old_h, old_w = prev_layer.out_shape[-2:]  # type: ignore
        pool_h, pool_w = self.pool_size
        new_h = (old_h - pool_h) // self.stride + 1
        new_w = (old_w - pool_w) // self.stride + 1

        self.input_shape = prev_layer.out_shape  # type: ignore
        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)  # type: ignore

    def forward(
        self, input: Union[PhiTensor, GammaTensor], *args: Any, **kwargs: Any
    ) -> Union[GammaTensor, PhiTensor]:
        # shape

        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size

        if self.out_shape is None:
            raise ValueError(
                "`self.out_shape` is None. Please check the out_shape is correctly initialized."
            )

        h_out, w_out = self.out_shape[-2:]

        # forward

        n, d, h, w = input.shape
        input_reshaped = input.reshape((n * d, 1, h, w))
        self.X_col = im2col_indices(
            input_reshaped, pool_h, pool_w, padding=0, stride=self.stride
        )

        self.max_idx = self.X_col._argmax(axis=0)
        outputs = self.X_col[self.max_idx, range(self.max_idx.size)]

        outputs = outputs.reshape((h_out, w_out, n, d))

        outputs = outputs.transpose((2, 3, 0, 1))

        return outputs

    def backward(
        self, pre_grad: Union[PhiTensor, GammaTensor], *args: Any, **kwargs: Any
    ) -> Union[GammaTensor, PhiTensor]:

        if self.input_shape is None:
            raise ValueError(
                "Input shape is None. \
                Please check if the input shape is correctly initialized from the prev_layer."
            )

        n, d, w, h = self.input_shape
        pool_h, pool_w = self.pool_size

        if self.X_col is None or self.max_idx is None:
            raise ValueError(
                "X_col or max_idx is not initialized. Please call .forward before calling .backward."
            )

        dX_col = self.X_col.zeros_like()

        dout_col = pre_grad.transpose((2, 3, 0, 1)).ravel()

        dout_col_size = np.prod(dout_col.shape)

        dX_col[self.max_idx, range(dout_col_size)] = dout_col  # type: ignore

        dX = col2im_indices(
            dX_col, (n * d, 1, h, w), pool_h, pool_w, padding=0, stride=self.stride
        )

        dX = dX.reshape(self.input_shape)

        return dX
