from ..autodp.phi_tensor import PhiTensor
from ..autodp.gamma_tensor import GammaTensor
from ..lazy_repeat_array import lazyrepeatarray as lra
from ...adp.data_subject_list import DataSubjectList as DSL

from typing import Union
import numpy as np


def Conv2d(image: Union[PhiTensor, GammaTensor], out_channels, kernel, padding=0, strides=1):
    """
    TODO:
    - Modify to work with DP Tensors
    - Modify to work with multiple images in a tensor
    """

    if not isinstance(padding, int):
        raise TypeError(f"Padding must be an integer, not type: {type(padding)}")
    kernel = np.flipud(np.fliplr(kernel))

    x_kernel, y_kernel = kernel.shape
    x_img, y_img = image.shape  # TODO: Needs to be modified if >1 image per tensor

    x_out = int(((x_img - x_kernel + 2 * padding)/strides) + 1)
    y_out = int(((y_img - y_kernel + 2 * padding)/strides) + 1)
    output = np.zeros(x_out, y_out)

    if padding != 0:
        padded_img = np.zeros((x_img + padding * 2, y_img + padding * 2))
        padded_img[padding: -padding, padding: -padding] = image
    else:
        padded_img = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - y_kernel:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - x_kernel:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * padded_img[x: x + x_kernel, y: y + y_kernel]).sum()
                except:
                    break

    return output

