# stdlib
from typing import Optional

# third party
import numpy as np

# relative
from ..autodp.phi_tensor import PhiTensor
from ..lazy_repeat_array import lazyrepeatarray as lra


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


def dp_leakyrelu(dp_tensor: PhiTensor, slope: float=0.01) -> PhiTensor:
    # TODO: Should we have an index in DSLs that corresponds to no data?

    gt = (dp_tensor.child > 0)
    return PhiTensor(
        child= gt * dp_tensor.child + (1 - gt) * dp_tensor.child * slope,
        data_subjects=dp_tensor.data_subjects,
        min_vals= lra(data=dp_tensor.min_vals.data * slope, shape=dp_tensor.min_vals.shape),
        max_vals= lra(data=dp_tensor.max_vals.data * slope, shape=dp_tensor.max_vals.shape),
    )


class leaky_ReLU(Activation):

    def __init__(self, slope=0.01):
        super(leaky_ReLU, self).__init__()
        self.slope = slope

    def forward(self, input_array: PhiTensor):
        # Last image that has been forward passed through this activation function
        self.last_forward = input_array
        return dp_leakyrelu(dp_tensor=input_array, slope=self.slope)

    def derivative(self, input_array: Optional[PhiTensor] = None):
        last_forward = input_array if input_array else self.last_forward
        res = np.ones(last_forward.shape)
        idx = last_forward <= 0
        res[idx.child] = self.slope

        return PhiTensor(child=res,
                         data_subjects=last_forward.data_subjects,
                         min_vals=last_forward.min_vals * 0,
                         max_vals=last_forward.max_vals * 1)
