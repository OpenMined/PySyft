import numpy as np
from ..autodp.phi_tensor import PhiTensor
from typing import Optional


def serial_strided_method(arr, sub_shape, stride):
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs


def serial_MaxPool(array: np.ndarray, kernel_size: int, stride: Optional[int]=None, pad: bool=False):
    m, n = array.shape[:2]
    if stride is None:
        stride = kernel_size
    _ceil = lambda x, y: x // y + 1
    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny - 1) * stride + kernel_size, (nx - 1) * stride + kernel_size) + array.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = array
    else:
        mat_pad = array[:(m - kernel_size) // stride * stride + kernel_size, :(n - kernel_size) // stride * stride + kernel_size, ...]
    view = serial_strided_method(mat_pad, (kernel_size, kernel_size), stride)
    return np.nanmax(view, axis=(2, 3))


def serial_AvgPool(array: np.ndarray, kernel_size: int, stride: Optional[int]=None, pad: bool=False):
    m, n = array.shape[:2]
    if stride is None:
        stride = kernel_size
    _ceil = lambda x, y: x // y + 1
    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny - 1) * stride + kernel_size, (nx - 1) * stride + kernel_size) + array.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = array
    else:
        mat_pad = array[:(m - kernel_size) // stride * stride + kernel_size, :(n - kernel_size) // stride * stride + kernel_size, ...]
    view = serial_strided_method(mat_pad, (kernel_size, kernel_size), stride)
    return np.nanmean(view, axis=(2, 3))


def vectorized_strided_method(arr, sub_shape, stride):
    sm, sh, sw, sc = arr.strides
    m, hi, wi, ci = arr.shape
    f1, f2 = sub_shape
    view_shape = (m, 1+(hi-f1)//stride, 1+(wi-f2)//stride, f1, f2, ci)
    strides = (sm, stride*sh, stride*sw, sh, sw, sc)
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs


def vectorized_MaxPool2d(array, kernel_size, stride=None, pad=False):
    m, hi, wi, ci = array.shape
    if stride is None:
        stride = kernel_size
    _ceil = lambda x, y: x//y + 1
    if pad:
        ny = _ceil(hi, stride)
        nx = _ceil(wi, stride)
        size = (m, (ny-1) * stride + kernel_size, (nx - 1) * stride + kernel_size, ci)
        mat_pad = np.full(size, 0)
        mat_pad[:, :hi, :wi, ...] = array
    else:
        mat_pad = array[:, :(hi - kernel_size) // stride * stride + kernel_size, :(wi - kernel_size) // stride * stride + kernel_size, ...]
    view = vectorized_strided_method(mat_pad, (kernel_size, kernel_size), stride)
    return np.nanmax(view, axis=(3, 4))


def vectorized_AvgPool2d(array, kernel_size, stride=None, pad=False):
    m, hi, wi, ci = array.shape
    if stride is None:
        stride = kernel_size
    _ceil = lambda x, y: x//y + 1
    if pad:
        ny = _ceil(hi, stride)
        nx = _ceil(wi, stride)
        size = (m, (ny-1) * stride + kernel_size, (nx - 1) * stride + kernel_size, ci)
        mat_pad = np.full(size, 0)
        mat_pad[:, :hi, :wi, ...] = array
    else:
        mat_pad = array[:, :(hi - kernel_size) // stride * stride + kernel_size, :(wi - kernel_size) // stride * stride + kernel_size, ...]
    view = vectorized_strided_method(mat_pad, (kernel_size, kernel_size), stride)
    return np.nanmean(view, axis=(3, 4))