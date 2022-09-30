# stdlib
from typing import Any
from typing import Sequence
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import NDArray

# relative
from ...adp.data_subject_list import DataSubjectArray
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor
from ..lazy_repeat_array import lazyrepeatarray


def dp_maximum(
    x: Union[PhiTensor, GammaTensor], y: Union[np.ndarray, PhiTensor, GammaTensor]
) -> Union[PhiTensor, GammaTensor]:
    # TODO: Make this work for GammaTensors
    x_data = x.child
    y_data = y.child if hasattr(y, "child") else y

    output = np.maximum(x_data, y_data)

    min_v, max_v = output.min(), output.max()
    dsl = x.data_subjects  # TODO: fix later
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
        min_v = lazyrepeatarray(
            data=np.log(input.min_vals.data.min()), shape=output.shape
        )
        max_v = lazyrepeatarray(
            data=np.log(input.max_vals.data.max()), shape=output.shape
        )
        dsl = input.data_subjects

        return PhiTensor(
            child=output,
            data_subjects=dsl,
            min_vals=min_v,
            max_vals=max_v,
        )
    elif isinstance(input, GammaTensor):
        output = np.log(input.child)
        # min_v, max_v = output.min(), output.max()  # These bounds are a function of private data
        min_v = lazyrepeatarray(
            data=np.log(input.min_vals.data.min()), shape=output.shape
        )
        max_v = lazyrepeatarray(
            data=np.log(input.max_vals.data.max()), shape=output.shape
        )
        dsl = input.data_subjects

        return GammaTensor(
            child=output,
            data_subjects=dsl,
            min_vals=min_v,
            max_vals=max_v,
        )
    else:
        raise NotImplementedError(f"Undefined behaviour for type: {type(input)}")


def dp_zeros(shape: Tuple) -> Union[PhiTensor, GammaTensor]:
    """
    TODO: Passing in the shape seems unnecessary- it can be inferred from data_subjects.data_subjects_indexed.shape
    output = np.zeros_like(data_subjects.data_subjects_indexed)

    :param shape:
    :param data_subjects:
    :return:
    """
    output = np.zeros(shape)

    # Create Empty DataSubjectSet array
    data_subjects = np.broadcast_to(np.array([DataSubjectArray(set())]), output.shape)

    return GammaTensor(
        child=output,
        data_subjects=data_subjects,
        min_vals=lazyrepeatarray(0, shape),
        max_vals=lazyrepeatarray(1, shape),
    )


def dp_pad(
    input: Union[PhiTensor, GammaTensor],
    width: Union[int, Sequence],
    padding_mode: str = "reflect",
    **kwargs: Any,
) -> Union[PhiTensor, GammaTensor]:

    data = input.child
    output_data: NDArray = np.pad(data, width, mode=padding_mode, **kwargs)
    min_v = lazyrepeatarray(
        data=min(input.min_vals.data.min(), output_data.min()), shape=output_data.shape
    )
    max_v = lazyrepeatarray(
        data=min(input.max_vals.data.max(), output_data.max()), shape=output_data.shape
    )

    output_data_subjects: NDArray = np.pad(
        input.data_subjects, width, mode=padding_mode, **kwargs
    )

    if isinstance(input, PhiTensor):
        return PhiTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_v,
            max_vals=max_v,
        )
    elif isinstance(input, GammaTensor):
        return GammaTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_v,
            max_vals=max_v,
        )
    else:
        raise NotImplementedError(
            f"Padding is not implemented for Input Type: {type(input)}"
        )


def dp_add_at(
    a: Union[PhiTensor, GammaTensor], indices: Tuple, b: Union[PhiTensor, GammaTensor]
) -> Union[PhiTensor, GammaTensor]:
    data_a = a.child
    data_b = b.child

    data_subject_a = a.data_subjects
    # data_subject_b = b.data_subjects

    np.add.at(data_a, indices, data_b)

    # TODO: @Shubham Find np.add.at can be implemented at data_subject_level
    # data_subject_a[indices] = data_subject_a[indices] + data_subject_b
    # The above is an alternate to np.add.at
    # np.add.at(data_subject_a, indices, data_subject_b)

    if isinstance(a, PhiTensor):
        return PhiTensor(
            child=data_a,
            data_subjects=data_subject_a,
            min_vals=data_a.min(),
            max_vals=data_a.max(),
        )
    else:
        return GammaTensor(
            child=data_a,
            data_subjects=data_subject_a,
            min_vals=lazyrepeatarray(data_a.min(), shape=data_a.shape),
            max_vals=lazyrepeatarray(data_a.min(), shape=data_a.shape),
        )


def get_im2col_indices(
    x_shape: Tuple[int, ...],
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    if (H + 2 * padding - field_height) % stride != 0:
        raise ValueError(
            f"(H + 2 * padding - field_height) H:{H} , padding:{padding} , field_height:{field_height} \
        should be a multiple of stride:{stride}"
        )
    if (W + 2 * padding - field_height) % stride != 0:
        raise ValueError(
            f"(W + 2 * padding - field_height) W:{W} , padding:{padding} , field_height:{field_height} \
        should be a multiple of stride:{stride}"
        )
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))


def col2im_indices(
    cols: Union[PhiTensor, GammaTensor],
    x_shape: Tuple[int, ...],
    field_height: int = 3,
    field_width: int = 3,
    padding: int = 1,
    stride: int = 1,
) -> Union[PhiTensor, GammaTensor]:
    """An implementation of col2im based on fancy indexing and np.add.at"""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = dp_zeros(shape=(N, C, H_padded, W_padded))

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape((C * field_height * field_width, -1, N))
    cols_reshaped = cols_reshaped.transpose((2, 0, 1))
    x_padded = dp_add_at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def im2col_indices(
    x: Union[PhiTensor, GammaTensor],
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
) -> Union[PhiTensor, GammaTensor]:
    """An implementation of im2col based on some fancy indexing"""
    # Zero-pad the input
    p = padding

    width: Tuple[Any, ...]
    if len(x.shape) == 4:
        width = ((0, 0), (0, 0), (p, p), (p, p))
    elif len(x.shape) == 3:
        width = ((0, 0), (p, p), (p, p))
    elif len(x.shape) == 2:
        width = ((p, p), (p, p))
    else:
        raise NotImplementedError
    x_padded = dp_pad(x, width)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose((1, 2, 0)).reshape((field_height * field_width * C, -1))

    # Not sure why this happens but sometimes this gets a shape of (n, -1)
    if cols.min_vals.shape != cols.shape:
        cols.min_vals.shape = cols.shape
        cols.max_vals.shape = cols.shape
    return cols
