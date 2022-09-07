# stdlib
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# relative
from ...common.serde.serializable import serializable
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor


@serializable(recursive_serde=True)
class Activation(object):
    """Base class for activations."""

    __attr_allowlist__: Tuple[str, ...] = ("last_forward",)

    def __init__(self) -> None:
        self.last_forward: Optional[Union[PhiTensor, GammaTensor]] = None

    def forward(
        self, input: Union[PhiTensor, GammaTensor]
    ) -> Union[PhiTensor, GammaTensor]:
        """Forward Step.

        Args:
            input (PhiTensor or GammaTensor): the input matrix
        """

        raise NotImplementedError

    def derivative(
        self, input: Optional[Union[PhiTensor, GammaTensor]] = None
    ) -> Union[PhiTensor, GammaTensor]:
        """Backward Step.

        _extended_summary_

        Args:
            input (Optional[PhiTensor, GammaTensor], optional): If provide `input`,
            this function will not use `last_forward`. Defaults to None.
        """

        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__


@serializable(recursive_serde=True)
class leaky_ReLU(Activation):
    __attr_allowlist__ = (
        "slope",
        "last_forward",
    )

    def __init__(self, slope: float = 0.01) -> None:
        super(leaky_ReLU, self).__init__()
        self.slope = slope

    def forward(
        self, input_array: Union[PhiTensor, GammaTensor]
    ) -> Union[PhiTensor, GammaTensor]:
        # Last image that has been forward passed through this activation function
        self.last_forward = input_array

        gt = input_array > 0

        return input_array * gt + input_array * ((gt * -1) + 1) * self.slope

    def derivative(
        self, input_array: Optional[Union[PhiTensor, GammaTensor]] = None
    ) -> Union[PhiTensor, GammaTensor]:
        last_forward = input_array if input_array else self.last_forward

        if last_forward is None:
            raise ValueError(
                "last forward layer is None. Please provide an input to the derivative method."
            )

        res = (last_forward > 0).child * 1 + (last_forward <= 0).child * self.slope

        if isinstance(input_array, PhiTensor):
            return PhiTensor(
                child=res,
                data_subjects=last_forward.data_subjects,
                min_vals=last_forward.min_vals * 0,
                max_vals=last_forward.max_vals * 1,
            )
        elif isinstance(input_array, GammaTensor):
            return GammaTensor(
                child=res,
                data_subjects=last_forward.data_subjects,
                min_vals=last_forward.min_vals * 0,
                max_vals=last_forward.max_vals * 1,
            )
        else:
            raise NotImplementedError(
                f"Undefined behavior for type {type(input_array)}"
            )


def get(activation_name: Optional[str]) -> Optional[Activation]:

    activation_name_map: Dict[str, Type[Activation]] = {
        "leaky_relu": leaky_ReLU,
    }

    if activation_name is None:
        return None
    elif isinstance(activation_name, str):
        if activation_name not in activation_name_map:
            raise NotImplementedError(
                f"Activation {activation_name} is not current supported in PySyft."
            )
        else:
            activation_func = activation_name_map[activation_name]
            return activation_func()
    else:
        raise TypeError(
            f"Could not interpret activation function identifier: {activation_name}"
        )
