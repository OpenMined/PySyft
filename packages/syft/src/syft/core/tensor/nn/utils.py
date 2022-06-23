import numpy as np
from ...adp.data_subject_list import DataSubjectList
from ..autodp.phi_tensor import PhiTensor


def dp_maximum(x, y):
    x_data = x.child
    y_data = y.child if hasattr(y, "child") else y

    output = np.maximum(x_data, y_data)
    min_v, max_v = output.min(), output.max()
    dsl = DataSubjectList(
        one_hot_lookup=x.data_subjects.one_hot_lookup,
        data_subjects_indexed=np.zeros_like(output)
    )
    return PhiTensor(
        child=output,
        data_subjects=dsl,
        min_vals=min_v,
        max_vals=max_v,
    )


def dp_log(input: PhiTensor):
    data = input.child

    output = np.log(data)
    min_v, max_v = output.min(), output.max()
    dsl = DataSubjectList(
        one_hot_lookup=input.data_subjects.one_hot_lookup,
        data_subjects_indexed=np.zeros_like(output)
    )
    return PhiTensor(
        child=output,
        data_subjects=dsl,
        min_vals=min_v,
        max_vals=max_v,
    )
