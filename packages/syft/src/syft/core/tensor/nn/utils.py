from typing import Tuple
from typing import Union
import numpy as np
from ...adp.data_subject_list import DataSubjectList
from ..autodp.phi_tensor import PhiTensor
from ..autodp.gamma_tensor import GammaTensor
from ..lazy_repeat_array import lazyrepeatarray


def dp_maximum(x: Union[np.ndarray, PhiTensor, GammaTensor], y: Union[np.ndarray, PhiTensor, GammaTensor]
               ) -> Union[PhiTensor, GammaTensor]:
    # TODO: Make this work for GammaTensors
    x_data = x.child
    y_data = y.child if hasattr(y, "child") else y

    output = np.maximum(x_data, y_data)


    """
    WORK IN PROGRESS- getting indices for data subjects
    
    # 0 means that x_data has highest value there, 1 means y_data has highest value there
    array_with_max = np.argmax(np.dstack((x_data, y_data)), axis=-1)

    # TODO: Can we do the below in just 1 function call/iteration instead of 2?
    x_max_ds = np.transpose(array_with_max.nonzero())  # coordinates where x supplies the max value
    y_max_ds = np.transpose((array_with_max == 1).nonzero())  # coordinates where y supplies the max value
    """

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
