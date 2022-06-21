from ..autodp.phi_tensor import PhiTensor
from ...adp.data_subject_list import DataSubjectList

from torch import Tensor
from torch import nn


def Linear(image: PhiTensor, in_features: int, out_features: int, bias=True) -> PhiTensor:
    linear_layer = nn.Linear(in_features, out_features, bias=bias)
    data = linear_layer(Tensor(image.child.decode())).detach().numpy()
    minv = linear_layer(Tensor(image.ones_like() * image.min_vals.data)).detach().numpy()
    maxv = linear_layer(Tensor(image.ones_like() * image.max_vals.data)).detach().numpy()

    return PhiTensor(
        child=data,
        data_subjects=DataSubjectList(
            one_hot_lookup=image.data_subjects.one_hot_lookup,
            data_subjects_indexed=image.data_subjects.zeros_like(data)
        ),
        min_vals=minv,
        max_vals=maxv
    )
