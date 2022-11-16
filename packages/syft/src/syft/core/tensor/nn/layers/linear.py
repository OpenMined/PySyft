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
from ..initializations import XavierInitialization
from .base import Layer


@serializable(recursive_serde=True)
class Linear(Layer):
    __attr_allowlist__ = (
        "n_out",
        "n_in",
        "out_shape",
        "W",
        "b",
        "dW",
        "db",
        "last_input",
        "init",
        "input_shape",
        "out_shape",
    )

    def __init__(self, n_out: int, n_in: Optional[int] = None) -> None:
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (n_out,)

        self.W: Optional[Union[NDArray, PhiTensor, GammaTensor]] = None
        self.b: Optional[Union[NDArray, PhiTensor, GammaTensor]] = None
        self.dW: Optional[Union[NDArray, PhiTensor, GammaTensor]] = None
        self.db: Optional[Union[NDArray, PhiTensor, GammaTensor]] = None
        self.last_input: Optional[Union[PhiTensor, GammaTensor]] = None
        self.init = XavierInitialization()

    def connect_to(self, prev_layer: Optional[Layer] = None) -> None:
        if prev_layer is None:
            if self.n_in is None:
                raise ValueError(
                    "`self.n_in` cannot be None since this is the first layer."
                )
            n_in = self.n_in
        else:
            if len(prev_layer.out_shape) != 2:  # type: ignore
                raise ValueError(
                    f"The out_shape should be of length 2. \
                    Current shape is : {len(prev_layer.out_shape)}"  # type: ignore
                )

            n_in = int(prev_layer.out_shape[-1])  # type: ignore

        self.W = self.init((n_in, self.n_out))
        self.b = np.zeros((self.n_out,))
        self.input_shape = (n_in,)
        self.out_shape = (self.n_out,)

    def forward(
        self,
        input: Union[PhiTensor, GammaTensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> Union[PhiTensor, GammaTensor]:
        self.last_input = input

        if self.W is None:
            raise ValueError(
                "`self.W` is None. Please check if self.W is correctly initialized."
            )

        return input.dot(self.W) + self.b

    def backward(
        self,
        pre_grad: Union[PhiTensor, GammaTensor],
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> Optional[Union[PhiTensor, GammaTensor]]:

        if self.last_input is None:
            raise ValueError(
                "last_input to the layer is None. \
                Please verify if the `self.last_input` is correctly initialized."
            )

        if self.W is None:
            raise ValueError(
                "self.W is None. \
                Please verify if the `self.W` is correctly initialized."
            )

        self.dW = self.last_input.T.dot(pre_grad)
        self.db = pre_grad.mean(axis=0)
        if not self.first_layer:
            return pre_grad.dot(self.W.T)

        return None

    @property
    def params(self) -> Tuple[Union[PhiTensor, GammaTensor, NDArray, None], ...]:
        return self.W, self.b

    @params.setter
    def params(
        self, new_params: Tuple[Union[PhiTensor, GammaTensor, NDArray, None], ...]
    ) -> None:
        if len(new_params) != 2:
            raise ValueError(
                f"Expected two values. Update params has length{len(new_params)}"
            )

        self.W, self.b = new_params

    @property
    def grads(self) -> Tuple[Union[PhiTensor, GammaTensor, NDArray, None], ...]:
        return self.dW, self.db
