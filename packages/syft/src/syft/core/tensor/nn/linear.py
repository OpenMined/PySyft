# third party
import numpy as np
from torch import Tensor
from torch import nn

# relative
from ...adp.data_subject_list import DataSubjectList
from ..autodp.phi_tensor import PhiTensor


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.func = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias
        )

    def forward(self, image: PhiTensor):
        image_asarray = image.child
        data = self.func(Tensor(image_asarray)).detach().numpy()
        minv = (
            self.func(Tensor(np.ones_like(image_asarray) * image.min_vals.data))
            .detach()
            .numpy()
        )
        maxv = (
            self.func(Tensor(np.ones_like(image_asarray) * image.max_vals.data))
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

    def parameters(self):
        return self.func.parameters()
