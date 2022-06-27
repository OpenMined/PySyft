# stdlib
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from ...adp.data_subject_list import DataSubjectList
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor
from ..lazy_repeat_array import lazyrepeatarray


def gamma_output(x: Union[np.ndarray, PhiTensor, GammaTensor], y: Union[np.ndarray, PhiTensor, GammaTensor]) -> bool:
    """ This function will tell you if the inputs to dp_maximum will be a GammaTensor.

    This will be the case if:
    - either x or y are gamma tensors
    - x, y are both phi tensors AND have different data subjects.

    TODO: This compares the full one_hot_lookup, not just the DS' who contributed the local maxima. Fix this?
    """
    inputs_are_gamma = any([isinstance(i, GammaTensor) for i in (x, y)])
    inputs_are_phi = all([isinstance(i, PhiTensor) for i in (x, y)])

    def phi_data_subjects_differ(x, y):
        if not isinstance(y, PhiTensor) or not isinstance(x, PhiTensor):
            return False

        if x.data_subjects.one_hot_lookup != y.data_subjects.one_hot_lookup:
            return True
        else:
            return False

    if inputs_are_gamma or (inputs_are_phi and phi_data_subjects_differ(x, y)):
        return True
    else:
        return False


def dp_maximum(x: Union[PhiTensor, GammaTensor], y: Union[np.ndarray, PhiTensor, GammaTensor]
               ) -> Union[PhiTensor, GammaTensor]:
    # TODO: Make this work for GammaTensors
    x_data = x.child
    y_data = y.child if hasattr(y, "child") else y

    output = np.maximum(x_data, y_data)

    if gamma_output(x, y):
        array_with_max = np.argmax(np.dstack((x_data, y_data)), axis=-1)
        x_max_ds = np.transpose(array_with_max.nonzero())
        y_max_ds = np.transpose((array_with_max == 1).nonzero())

        tensor_list = [x[tuple(i)] for i in x_max_ds]
        if isinstance(y, (PhiTensor, GammaTensor)):
            tensor_list += [y[tuple(i)] for i in y_max_ds]
        return GammaTensor.combine(tensor_list, output.shape)

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


def dp_log(input: Union[PhiTensor, GammaTensor]) -> Union[PhiTensor, GammaTensor]:
    if isinstance(input, PhiTensor):
        output = np.log(input.child)
        # min_v, max_v = output.min(), output.max()  # These bounds are a function of private data
        min_v = lazyrepeatarray(data=np.log(input.min_vals.data.min()), shape=output.shape)
        max_v = lazyrepeatarray(data=np.log(input.max_vals.data.max()), shape=output.shape)
        dsl = input.data_subjects

        return PhiTensor(
            child=output,
            data_subjects=dsl,
            min_vals=min_v,
            max_vals=max_v,
        )
    elif isinstance(input, GammaTensor):
        return input.log()  # TODO: this doesn't technically match np API so
    else:
        raise NotImplementedError(f"Undefined behaviour for type: {type(input)}")


def dp_zeros(shape: Tuple, data_subjects: DataSubjectList) -> Union[PhiTensor, GammaTensor]:
    """
    TODO: Passing in the shape seems unnecessary- it can be inferred from data_subjects.data_subjects_indexed.shape
    output = np.zeros_like(data_subjects.data_subjects_indexed)

    :param shape:
    :param data_subjects:
    :return:
    """

    output = np.zeros(shape)

    result = PhiTensor(
        child=output,
        data_subjects=DataSubjectList(
            one_hot_lookup=data_subjects.one_hot_lookup,
            data_subjects_indexed=np.zeros_like(output)  # This shouldn't matter b/c it will be replaced
        ),
        min_vals=output.min(),
        max_vals=output.max()
    )

    ds_count = len(data_subjects.one_hot_lookup)

    if ds_count == 1:
        return result
    elif ds_count > 1:
        return result.gamma  # TODO @Ishan: will the lack of a `gamma.func` here hurt us in any way?
    else:
        raise NotImplementedError("Zero or negative data subject behaviour undefined.")
