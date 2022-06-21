# third party
import numpy as np
from torch import Tensor
from torch import nn

# relative
from ...adp.data_subject_list import DataSubjectList
from ..autodp.phi_tensor import PhiTensor


def Linear(
    image: PhiTensor, in_features: int, out_features: int, bias: bool = True
) -> PhiTensor:
    linear_layer = nn.Linear(in_features, out_features, bias=bias)
    image_asarray = image.child.decode()
    data = linear_layer(Tensor(image_asarray)).detach().numpy()
    minv = (
        linear_layer(Tensor(np.ones_like(image_asarray) * image.min_vals.data))
        .detach()
        .numpy()
    )
    maxv = (
        linear_layer(Tensor(np.ones_like(image_asarray) * image.max_vals.data))
        .detach()
        .numpy()
    )

    return PhiTensor(
        child=data,
        data_subjects=DataSubjectList(
            one_hot_lookup=image.data_subjects.one_hot_lookup,
            data_subjects_indexed=np.zeros_like(data),
        ),
        min_vals=minv,
        max_vals=maxv,
    )
