"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
for all native python objects.
"""
from syft.workers import AbstractWorker
from typing import Tuple
import numpy


def _simplify_str(obj: str) -> tuple:
    return (obj.encode("utf-8"),)


def _detail_str(worker: AbstractWorker, str_tuple: tuple) -> str:
    return str_tuple[0].decode("utf-8")


# Range


def _simplify_range(my_range: range) -> Tuple[int, int, int]:
    """
    This function extracts the start, stop and step from the range.

    Args:
        my_range (range): a range object

    Returns:
        list: a list defining the range parameters [start, stop, step]

    Examples:

        range_parameters = _simplify_range(range(1, 3, 4))

        assert range_parameters == [1, 3, 4]

    """

    return (my_range.start, my_range.stop, my_range.step)


def _detail_range(worker: AbstractWorker, my_range_params: Tuple[int, int, int]) -> range:
    """
    This function extracts the start, stop and step from a tuple.

    Args:
        worker: the worker doing the deserialization (only here to standardise signature
            with other _detail functions)
        my_range_params (tuple): a tuple defining the range parameters [start, stop, step]

    Returns:
        range: a range object

    Examples:
        new_range = _detail_range([1, 3, 4])

        assert new_range == range(1, 3, 4)

    """

    return range(my_range_params[0], my_range_params[1], my_range_params[2])


def _simplify_ellipsis(e: Ellipsis) -> bytes:
    return b""


def _detail_ellipsis(worker: AbstractWorker, ellipsis: bytes) -> Ellipsis:
    return ...


def _simplify_slice(my_slice: slice) -> Tuple[int, int, int]:
    """
    This function creates a list that represents a slice.

    Args:
        my_slice (slice): a python slice

    Returns:
        tuple : a list holding the start, stop and step values

    Examples:

        slice_representation = _simplify_slice(slice(1,2,3))

    """
    return (my_slice.start, my_slice.stop, my_slice.step)


def _detail_slice(worker: AbstractWorker, my_slice: Tuple[int, int, int]) -> slice:
    """
    This function extracts the start, stop and step from a list.

    Args:
        my_slice (tuple): a list defining the slice parameters [start, stop, step]

    Returns:
        range: a range object

    Examples:
        new_range = _detail_range([1, 3, 4])

        assert new_range == range(1, 3, 4)

    """

    return slice(my_slice[0], my_slice[1], my_slice[2])


def _simplify_ndarray(my_array: numpy.ndarray) -> Tuple[bin, Tuple, str]:
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
    arr_bytes = my_array.tobytes()
    arr_shape = my_array.shape
    arr_dtype = my_array.dtype.name

    return (arr_bytes, arr_shape, arr_dtype)


def _detail_ndarray(
        worker: AbstractWorker, arr_representation: Tuple[bin, Tuple, str]
) -> numpy.ndarray:
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
    res = numpy.frombuffer(arr_representation[0], dtype=arr_representation[2]).reshape(
        arr_representation[1]
    )

    assert type(res) == numpy.ndarray

    return res
