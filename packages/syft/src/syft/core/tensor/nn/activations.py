# stdlib
from typing import Optional

# third party
import numpy as np

# relative
from ...common.serde.serializable import serializable
from ..autodp.phi_tensor import PhiTensor


@serializable(recursive_serde=True)
class Activation(object):
    """Base class for activations.

    """
    def __init__(self):
        self.last_forward = None

    def forward(self, input: PhiTensor):
        """Forward Step.

        Args:
            input (PhiTensor): the input matrix
        """

        raise NotImplementedError

    def derivative(self, input: Optional[PhiTensor]=None):
        """Backward Step.

        _extended_summary_

        Args:
            input (Optional[PhiTensor], optional): If provide `input`, this function will not use `last_forward`. Defaults to None.
        """

        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class leaky_ReLU(Activation):

    def __init__(self, slope=0.01):
        super(leaky_ReLU, self).__init__()
        self.slope = slope

    def forward(self, input_array: PhiTensor):
        # Last image that has been forward passed through this activation function
        self.last_forward = input_array

        gt = input_array > 0

        return gt * input_array + ((gt * -1) + 1) * input_array * self.slope

    def derivative(self, input_array: Optional[PhiTensor] = None):
        last_forward = input_array if input_array else self.last_forward
        res = np.ones(last_forward.shape)
        idx = last_forward <= 0
        res[idx.child] = self.slope

        return PhiTensor(child=res,
                         data_subjects=last_forward.data_subjects,
                         min_vals=last_forward.min_vals * 0,
                         max_vals=last_forward.max_vals * 1)
