# stdlib
from typing import Tuple
from typing import Union

# third party
import numpy as np
from torch import Tensor
from torch import nn

# relative
from ...adp.data_subject_list import DataSubjectList as DSL
from ..autodp.phi_tensor import PhiTensor


def np_to_torch(array: np.ndarray) -> Tensor:
    dims = len(array.shape)
    if dims == 3:
        return Tensor(array.reshape(1, *array.shape[::-1]))
    elif dims == 4:
        # TODO: Check this against data in a domain node
        return Tensor(array)
        # return Tensor(array.reshape(*array.shape[-2:], *array.shape[:-2]))
    else:
        raise NotImplementedError


def child_to_torch(dp_tensor: PhiTensor) -> Tensor:
    return Tensor(np_to_torch(dp_tensor.child.decode()))


def Conv2d(
    image: PhiTensor,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    bias: bool = True,
):
    # TODO: Figure out how to make the min/max val bounds with public data!
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    # TODO: This conversion is only required the first time and not after that.
    torch_tensor = child_to_torch(image)
    data = conv_layer(torch_tensor)
    data_array = data.detach().numpy()

    return PhiTensor(
        child=data_array,
        data_subjects=DSL(
            one_hot_lookup=image.data_subjects.one_hot_lookup,
            data_subjects_indexed=np.zeros_like(data_array),
        ),
        min_vals=data_array.min(),
        max_vals=data_array.max(),
    )


#
# def Conv2d(image: Union[PhiTensor, GammaTensor], out_channels, kernel, padding=0, strides=1):
#     """
#     TODO:
#     - Modify to work with DP Tensors
#     - Modify to work with multiple images in a tensor
#     """
#
#     if not isinstance(padding, int):
#         raise TypeError(f"Padding must be an integer, not type: {type(padding)}")
#     kernel = np.flipud(np.fliplr(kernel))
#
#     x_kernel, y_kernel = kernel.shape
#     x_img, y_img = image.shape  # TODO: Needs to be modified if >1 image per tensor
#
#     x_out = int(((x_img - x_kernel + 2 * padding)/strides) + 1)
#     y_out = int(((y_img - y_kernel + 2 * padding)/strides) + 1)
#     output = np.zeros(x_out, y_out)
#
#     if padding != 0:
#         padded_img = np.zeros((x_img + padding * 2, y_img + padding * 2))
#         padded_img[padding: -padding, padding: -padding] = image
#     else:
#         padded_img = image
#
#     for y in range(image.shape[1]):
#         if y > image.shape[1] - y_kernel:
#             break
#         if y % strides == 0:
#             for x in range(image.shape[0]):
#                 if x > image.shape[0] - x_kernel:
#                     break
#                 try:
#                     if x % strides == 0:
#                         output[x, y] = (kernel * padded_img[x: x + x_kernel, y: y + y_kernel]).sum()
#                 except:
#                     break
#
#     return output
