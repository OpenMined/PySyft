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

from syft.workers.abstract import AbstractWorker
from syft import serde


# Simplify/Detail Collections (list, set, tuple, etc.)


def _simplify_collection(my_collection: Collection) -> Tuple:
    """
    This function is designed to search a collection for any objects
    which may need to be simplified (i.e., torch tensors). It iterates
    through each object in the collection and calls _simplify on it. Finally,
    it returns the output as the tuple of simplified items of the input collection.
    This function is used to simplify list, set, and tuple. The reverse function,
    which undoes the functionality of this function is different for each of these types:
    _detail_collection_list, _detail_collection_set, _detail_collection_tuple.

    Args:
        my_collection (Collection): a collection of python objects

    Returns:
        Tuple: a tuple with simplified objects.

    """

    # Step 0: initialize empty list
    pieces = list()

    # Step 1: serialize each part of the collection
    for part in my_collection:
        pieces.append(serde._simplify(part))

    # Step 2: return serialization as tuple of simplified items
    return tuple(pieces)


def _detail_collection_list(worker: AbstractWorker, my_collection: Tuple) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a tuple of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_collection (Tuple): a tuple of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        detailed = serde._detail(worker, part)
        if isinstance(detailed, bytes):
            detailed = detailed.decode("utf-8")
        pieces.append(detailed)

    return pieces


def _detail_collection_set(worker: AbstractWorker, my_collection: Tuple) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a tuple of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_collection (Tuple): a tuple of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        detailed = serde._detail(worker, part)
        if isinstance(detailed, bytes):
            detailed = detailed.decode("utf-8")
        pieces.append(detailed)
    return set(pieces)


def _detail_collection_tuple(worker: AbstractWorker, my_tuple: Tuple) -> Tuple:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a tuple of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.
    This is only applicable to tuples. They need special handling because
    `msgpack` is encoding a tuple as a list.

    Args:
        worker: the worker doing the deserialization
        my_tuple (Tuple): a collection of simple python objects (including binary).

    Returns:
        tuple: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_tuple:
        pieces.append(serde._detail(worker, part))

    return tuple(pieces)


def _simplify_dictionary(my_dict: Dict) -> Tuple:
    """
    This function is designed to search a dict for any objects
    which may need to be simplified (i.e., torch tensors). It iterates
    through each key, value in the dict and calls _simplify on it. Finally,
    it returns the output tuple of tuples containing key/value pairs. The
    reverse function to this function is _detail_dictionary, which undoes
    the functionality of this function.

    Args:
        my_dict: A dictionary of python objects.

    Returns:
        Tuple: Tuple containing tuples of simplified key/value pairs from the
            input dictionary.

    """
    pieces = list()
    # for dictionaries we want to simplify both the key and the value
    for key, value in my_dict.items():
        pieces.append((serde._simplify(key), serde._simplify(value)))

    return tuple(pieces)


def _detail_dictionary(worker: AbstractWorker, my_dict: Tuple) -> Dict:
    """
    This function is designed to operate in the opposite direction of
    _simplify_dictionary. It takes a dictionary of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_dict (Tuple): a simplified dictionary of simple python objects (including binary).

    Returns:
        Dict: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """
    pieces = {}
    # for dictionaries we want to detail both the key and the value
    for key, value in my_dict:
        detailed_key = serde._detail(worker, key)
        if isinstance(detailed_key, bytes):
            detailed_key = detailed_key.decode("utf-8")

        detailed_value = serde._detail(worker, value)
        if isinstance(detailed_value, bytes):
            detailed_value = detailed_value.decode("utf-8")
        pieces[detailed_key] = detailed_value

    return pieces


# Simplify/Detail native types


def _simplify_str(obj: str) -> tuple:
    return (obj.encode("utf-8"),)


def _detail_str(worker: AbstractWorker, str_tuple: tuple) -> str:
    return str_tuple[0].decode("utf-8")


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


#   Numpy array


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


def _simplify_numpy_number(
    numpy_nb: Union[numpy.int32, numpy.int64, numpy.float32, numpy.float64]
) -> Tuple[bin, Tuple]:
    """
    This function gets the byte representation of the numpy number
        and stores the dtype for reconstruction

    Args:
        numpy_nb (e.g numpy.float64): a numpy number

    Returns:
        list: a list holding the byte representation, dtype of the numpy number

    Examples:

        np_representation = _simplify_numpy_number(numpy.float64(2.3)))

    """
    nb_bytes = numpy_nb.tobytes()
    nb_dtype = numpy_nb.dtype.name

    return (nb_bytes, nb_dtype)


def _detail_numpy_number(
    worker: AbstractWorker, nb_representation: Tuple[bin, Tuple, str]
) -> Union[numpy.int32, numpy.int64, numpy.float32, numpy.float64]:
    """
    This function reconstruct a numpy number from it's byte data, dtype
        by first loading the byte data with the appropiate dtype

    Args:
        worker: the worker doing the deserialization
        np_representation (tuple): a tuple holding the byte representation
        and dtype of the numpy number

    Returns:
        numpy.float or numpy.int: a numpy number

    Examples:
        nb = _detail_numpy_number(nb_representation)

    """
    nb = numpy.frombuffer(nb_representation[0], dtype=nb_representation[1])[0]

    assert type(nb) in [numpy.float32, numpy.float64, numpy.int32, numpy.int64]

    return nb


# Maps a type to a tuple containing its simplifier and detailer function
# IMPORTANT: keep this structure sorted A-Z (by type name)
MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        dict: (_simplify_dictionary, _detail_dictionary),
        list: (_simplify_collection, _detail_collection_list),
        range: (_simplify_range, _detail_range),
        set: (_simplify_collection, _detail_collection_set),
        slice: (_simplify_slice, _detail_slice),
        str: (_simplify_str, _detail_str),
        tuple: (_simplify_collection, _detail_collection_tuple),
        type(Ellipsis): (_simplify_ellipsis, _detail_ellipsis),
        numpy.ndarray: (_simplify_ndarray, _detail_ndarray),
        numpy.float32: (_simplify_numpy_number, _detail_numpy_number),
        numpy.float64: (_simplify_numpy_number, _detail_numpy_number),
        numpy.int32: (_simplify_numpy_number, _detail_numpy_number),
        numpy.int64: (_simplify_numpy_number, _detail_numpy_number),
    }
)
