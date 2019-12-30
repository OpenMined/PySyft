"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
for all native python objects.
"""
from collections import OrderedDict
from typing import Collection
from typing import Dict
from typing import Union
from typing import Tuple


import numpy

from syft import ndarray as SyftNdarray
from syft.workers.abstract import AbstractWorker
from syft.serde.native_serde import _simplify_ndarray, _detail_ndarray


#   Numpy array
def _simplify_syft_ndarray(worker: AbstractWorker, my_array: SyftNdarray) -> Tuple[bin, Tuple, str]:
    """
    This function gets the byte representation of the array
        and stores the dtype and shape for reconstruction

    Args:
        my_array (numpy.ndarray): a numpy array

    Returns:
        list: a list holding the byte representation, shape and dtype of the array

    Examples:

        arr_representation = _simplify_ndarray(numpy.random.random([1000, 1000])))

    """
    (arr_bytes, arr_shape, arr_dtype) = _simplify_ndarray(worker, my_array)

    return (arr_bytes, arr_shape, arr_dtype)


def _detail_syft_ndarray(
    worker: AbstractWorker, arr_representation: Tuple[bin, Tuple, str]
) -> SyftNdarray:
    """
    This function reconstruct a numpy array from it's byte data, the shape and the dtype
        by first loading the byte data with the appropiate dtype and then reshaping it into the
        original shape

    Args:
        worker: the worker doing the deserialization
        arr_representation (tuple): a tuple holding the byte representation, shape
        and dtype of the array

    Returns:
        numpy.ndarray: a numpy array

    Examples:
        arr = _detail_ndarray(arr_representation)

    """
    res = _detail_ndarray(worker, arr_representation).view(SyftNdarray)

    assert type(res) == SyftNdarray

    return res


# Maps a type to a tuple containing its simplifier and detailer function
# IMPORTANT: serialization constants for these objects need to be defined in `proto.json` file
# in https://github.com/OpenMined/proto
MAP_SYFT_NDARRAY_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        SyftNdarray: (_simplify_syft_ndarray, _detail_syft_ndarray),
    }
)
