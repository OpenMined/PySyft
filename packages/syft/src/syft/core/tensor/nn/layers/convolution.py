# stdlib
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import NDArray

# relative
from .. import activations
from ....common.serde.serializable import serializable
from ...autodp.gamma_tensor import GammaTensor
from ...autodp.phi_tensor import PhiTensor
from ..initializations import XavierInitialization
from ..utils import col2im_indices
from ..utils import im2col_indices
from .base import Layer


@serializable(recursive_serde=True)
class Convolution(Layer):
    """
    If this is the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does NOT include the sample axis, N.),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
    """

    __attr_allowlist__ = [
        "nb_filter",
        "filter_size",
        "input_shape",
        "stride",
        "padding",
        "W",
        "b",
        "dW",
        "db",
        "out_shape",
        "last_output",
        "last_input",
        "init",
        "activation",
    ]

    def __init__(
        self,
        nb_filter: int,
        filter_size: Union[int, Tuple[int, int]],
        input_shape: Optional[Tuple] = None,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[str] = None,
    ):
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride
        self.padding = padding
        self.name = "Conv"

        self.W: Optional[Union[NDArray, PhiTensor, GammaTensor]] = None
        self.dW: Optional[Union[PhiTensor, GammaTensor]] = None
        self.b: Optional[Union[NDArray, PhiTensor, GammaTensor]] = None
        self.db: Optional[Union[PhiTensor, GammaTensor]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None
        self.last_output: Optional[Union[PhiTensor, GammaTensor]] = None
        self.last_input: Optional[Union[PhiTensor, GammaTensor]] = None

        self.init = XavierInitialization()
        self.activation = activations.get(activation)

    def connect_to(self, prev_layer: Optional[Layer] = None) -> None:
        if prev_layer is None:
            if self.input_shape is None:
                raise ValueError(
                    "`input_shape` cannot be None. \
                    Please specify the input shape, since this is the first layer."
                )
            input_shape = self.input_shape
        else:
            input_shape = prev_layer.out_shape  # type: ignore

        if len(input_shape) != 4:
            raise ValueError(
                "Input shape to the layer should be of the format: (N, C, H, W)."
            )

        nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        if isinstance(self.filter_size, tuple):
            filter_height, filter_width = self.filter_size
        elif isinstance(self.filter_size, int):
            filter_height = filter_width = self.filter_size
        else:
            raise TypeError(
                "Argument `filter_size` should either to be Tuple[int, int] or an integer."
            )

        height = (pre_height - filter_height + 2 * self.padding) // self.stride + 1
        width = (pre_width - filter_width + 2 * self.padding) // self.stride + 1

        # output shape
        self.out_shape = (nb_batch, self.nb_filter, height, width)

        # filters
        self.W = self.init((self.nb_filter, pre_nb_filter, filter_height, filter_width))
        self.b = np.zeros((self.nb_filter,))

    def forward(
        self,
        input: Union[PhiTensor, GammaTensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> Union[PhiTensor, GammaTensor]:

        self.last_input = input
        self.input_shape = input.shape

        if self.W is None or self.b is None:
            raise ValueError(
                """Weight or bias for the layer cannot be None.
                Please initialize the weight (self.W) and bias (self.b) for the layer."""
            )

        # N, C, H, W
        n_filters, _, h_filter, w_filter = self.W.shape

        # Input Shape -> N, C, H, W
        n_x, _, _, _ = input.shape

        if self.out_shape is None:
            raise ValueError(
                "`self.out_shape` cannot be None. Please check if the output_shape to the layer is correctly defined."
            )

        _, _, h_out, w_out = self.out_shape

        self.X_col: Union[GammaTensor, PhiTensor] = im2col_indices(
            input, h_filter, w_filter, padding=self.padding, stride=self.stride
        )

        W_col = self.W.reshape((n_filters, -1))
        out = (
            self.X_col.T @ W_col.T + self.b
        )  # Transpose is required here because W_col is numpy array
        out = out.reshape((n_filters, h_out, w_out, n_x))
        out = out.transpose((3, 0, 1, 2))

        self.last_output = (
            self.activation.forward(out) if self.activation is not None else out
        )

        return out

    def backward(
        self,
        pre_grad: Union[PhiTensor, GammaTensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> Union[PhiTensor, GammaTensor]:

        if self.W is None or self.b is None:
            raise ValueError(
                """Weight or bias for the layer cannot be None.
                Please initialize the weight (self.W) and bias (self.b) for the layer."""
            )

        # N, C, H, W
        n_filter, _, h_filter, w_filter = self.W.shape

        pre_grads = (
            self.activation.derivative(pre_grad)
            if self.activation is not None
            else pre_grad
        )

        db = pre_grads.sum(axis=(0, 2, 3))

        self.db = db.reshape((n_filter,))

        pre_grads_reshaped = pre_grads.transpose((1, 2, 3, 0))
        pre_grads_reshaped = pre_grads_reshaped.reshape((n_filter, -1))
        pre_grads_reshaped.min_vals.shape = (
            pre_grads_reshaped.max_vals.shape
        ) = pre_grads_reshaped.shape
        dW = pre_grads_reshaped @ self.X_col.T
        self.dW = dW.reshape((self.W.shape))

        W_reshape = self.W.reshape((n_filter, -1))
        dX_col = pre_grads_reshaped.T @ W_reshape

        if self.input_shape is None:
            raise ValueError(
                "Input shape to the layer cannot be None. \
                Please check if the input shape has been initialized ."
            )
        dX = col2im_indices(
            dX_col,
            self.input_shape,
            h_filter,
            w_filter,
            padding=self.padding,
            stride=self.stride,
        )
        return dX

    @property
    def params(self) -> Tuple[Union[PhiTensor, GammaTensor, NDArray, None], ...]:
        return self.W, self.b

    @params.setter
    def params(
        self, new_params: Tuple[Union[PhiTensor, GammaTensor, NDArray], ...]
    ) -> None:

        if len(new_params) != 2:
            raise ValueError(
                f"Expected two values. Update params has length {len(new_params)}"
            )
        self.W, self.b = new_params

    @property
    def grads(self) -> Tuple[Union[PhiTensor, GammaTensor, NDArray, None], ...]:
        return self.dW, self.db
