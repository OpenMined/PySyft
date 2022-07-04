# stdlib
from typing import Sequence
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from ...adp.data_subject_list import DataSubjectList
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor
from ..lazy_repeat_array import lazyrepeatarray


def gamma_output(
    x: Union[np.ndarray, PhiTensor, GammaTensor],
    y: Union[np.ndarray, PhiTensor, GammaTensor],
) -> bool:
    """This function will tell you if the inputs to dp_maximum will be a GammaTensor.

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


def dp_maximum(
    x: Union[PhiTensor, GammaTensor], y: Union[np.ndarray, PhiTensor, GammaTensor]
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
        data_subjects_indexed=np.zeros_like(output),
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
        return input.log()  # TODO: this doesn't technically match np API so
    else:
        raise NotImplementedError(f"Undefined behaviour for type: {type(input)}")


def dp_zeros(
    shape: Tuple, data_subjects: DataSubjectList
) -> Union[PhiTensor, GammaTensor]:
    """
    TODO: Passing in the shape seems unnecessary- it can be inferred from data_subjects.data_subjects_indexed.shape
    output = np.zeros_like(data_subjects.data_subjects_indexed)

    :param shape:
    :param data_subjects:
    :return:
    """
    output = np.zeros(shape)
    ds_count = len(data_subjects.one_hot_lookup)

    if ds_count == 1:

        return PhiTensor(
            child=output,
            data_subjects=DataSubjectList(
                one_hot_lookup=data_subjects.one_hot_lookup,
                data_subjects_indexed=np.zeros_like(
                    output
                ),  # This shouldn't matter b/c it will be replaced
            ),
            min_vals=output.min(),
            max_vals=output.max(),
        )
    elif ds_count > 1:
        # TODO @Ishan: will the lack of a `gamma.func` here hurt us in any way?
        return GammaTensor(
            child=output,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(0, shape),
            max_vals=lazyrepeatarray(1, shape),
        )
    else:
        raise NotImplementedError("Zero or negative data subject behaviour undefined.")


def dp_pad(
    input: Union[PhiTensor, GammaTensor], width, padding_mode="reflect", **kwargs
):

    data = input.child
    output_data: Sequence = np.pad(data, width, mode=padding_mode, **kwargs)
    min_v = lazyrepeatarray(
        data=min(input.min_vals.data.min(), output_data.min()), shape=output_data.shape
    )
    max_v = lazyrepeatarray(
        data=min(input.max_vals.data.max(), output_data.max()), shape=output_data.shape
    )
    if isinstance(input, GammaTensor):
        dsl_width = [(0, 0)] + list(width)
    else:
        dsl_width = width
    print("dsi_shape, width, mode", input.data_subjects.shape, width, padding_mode)
    output_dsi = np.pad(
        input.data_subjects.data_subjects_indexed,
        pad_width=dsl_width,
        mode=padding_mode,
    )

    output_data_subjects = DataSubjectList(
        one_hot_lookup=input.data_subjects.one_hot_lookup,
        data_subjects_indexed=output_dsi,
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


def dp_add_at(a: PhiTensor, indices: Tuple, b: PhiTensor):
    data_a = a.child
    data_b = b.child

    np.add.at(data_a, indices, data_b)

    return PhiTensor(
        child=data_a,
        data_subjects=a.data_subjects,
        min_vals=data_a.min(),
        max_vals=data_a.max(),
    )


def get_im2col_indices(
    x_shape: Tuple,
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
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
    cols: PhiTensor,
    x_shape: Tuple,
    field_height: int = 3,
    field_width: int = 3,
    padding: int = 1,
    stride: int = 1,
):
    """An implementation of col2im based on fancy indexing and np.add.at"""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = dp_zeros((N, C, H_padded, W_padded), data_subjects=cols.data_subjects)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape((C * field_height * field_width, -1, N))
    cols_reshaped = cols_reshaped.transpose((2, 0, 1))
    x_padded = dp_add_at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def im2col_indices(
    x: PhiTensor, field_height: int, field_width: int, padding: int = 1, stride: int = 1
):
    """An implementation of im2col based on some fancy indexing"""
    # Zero-pad the input
    p = padding
    print("shapes before padding:, ", x.shape, x.data_subjects.shape)

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

    # return k, i, j, x_padded
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose((1, 2, 0)).reshape((field_height * field_width * C, -1))

    # Not sure why this happens but sometimes this gets a shape of (n, -1)
    if cols.min_vals.shape != cols.shape:
        cols.min_vals.shape = cols.shape
        cols.max_vals.shape = cols.shape
    return cols
