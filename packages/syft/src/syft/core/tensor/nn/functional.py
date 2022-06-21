# third party
from torch import Tensor
from torch import nn

# relative
from ..autodp.phi_tensor import PhiTensor


def leaky_relu(input: PhiTensor, negative_slope: float = 0.01) -> PhiTensor:

    data = nn.functional.leaky_relu(Tensor(input.child.decode()), negative_slope)
    data_as_numpy = data.detach().numpy()

    return PhiTensor(
        child=data_as_numpy,
        data_subjects=input.data_subjects,
        min_vals=data_as_numpy.min(),
        max_vals=data_as_numpy.max(),
    )
