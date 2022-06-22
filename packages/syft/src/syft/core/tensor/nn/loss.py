# stdlib
from typing import Optional

# third party
import numpy as np
from torch import Tensor
from torch import nn

# relative
from ...adp.data_subject_list import DataSubjectList as DSL
from ..autodp.phi_tensor import PhiTensor


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.func = nn.CrossEntropyLoss(
            self.weight,
            self.size_average,
            self.ignore_index,
            self.reduce,
            self.reduction,
        )

    def forward(self, input: PhiTensor, target: PhiTensor):
        input_asarray = input.child
        target_asarray = target.child

        data = (
            self.func(Tensor(input_asarray), Tensor(target_asarray).long())
            .detach()
            .numpy()
        )

        minv = (
            self.func(
                Tensor(np.ones_like(input_asarray) * input.min_vals.data),
                Tensor(np.ones_like(target_asarray) * target.min_vals.data).long(),
            )
            .detach()
            .numpy()
        )
        maxv = self.func(
            Tensor(np.ones_like(input_asarray) * input.max_vals.data),
            Tensor(np.ones_like(target_asarray) * target.max_vals.data).long(),
        )

        return PhiTensor(
            child=data,
            data_subjects=DSL(
                one_hot_lookup=input.data_subjects.one_hot_lookup,
                data_subjects_indexed=np.zeros_like(data),
            ),
            min_vals=minv,
            max_vals=maxv,
        )
