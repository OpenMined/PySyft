"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
for all native python objects.
"""
from collections import OrderedDict
from typing import Collection
from typing import Dict
from typing import Tuple

import numpy

from syft.workers import AbstractWorker
from syft import serde


# Simplify/Detail Collections (list, set, tuple, etc.)


def _simplify_collection(my_collection: Collection) -> Collection:
    """
    This function is designed to search a collection for any objects
    which may need to be simplified (i.e., torch tensors). It iterates
    through each object in the collection and calls _simplify on it. Finally,
    it returns the output collection as the same type as the input collection
    so that the consuming serialization step knows the correct type info. The
    reverse function to this function is _detail_collection, which undoes
    the functionality of this function.

    Args:
        my_collection (Collection): a collection of python objects

    Returns:
        Collection: a collection of the same type as the input of simplified
            objects.

    """

    # Step 0: get collection type for later use and itialize empty list
    my_type = type(my_collection)
    pieces = list()

    # Step 1: serialize each part of the collection
    for part in my_collection:
        pieces.append(serde._simplify(part))

    # Step 2: convert back to original type and return serialization
    if my_type == set:
        return pieces

    return tuple(pieces)


def _detail_collection_list(worker: AbstractWorker, my_collection: Collection) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a collection of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_collection (Collection): a collection of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        try:
            pieces.append(
                serde._detail(worker, part).decode("utf-8")
            )  # transform bytes back to string
        except AttributeError:
            pieces.append(serde._detail(worker, part))

    return pieces


def _detail_collection_set(worker: AbstractWorker, my_collection: Collection) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a collection of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_collection (Collection): a collection of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        try:
            pieces.append(
                serde._detail(worker, part).decode("utf-8")
            )  # transform bytes back to string
        except AttributeError:
            pieces.append(serde._detail(worker, part))
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


def _simplify_dictionary(my_dict: Dict) -> Dict:
    """
    This function is designed to search a dict for any objects
    which may need to be simplified (i.e., torch tensors). It iterates
    through each key, value in the dict and calls _simplify on it. Finally,
    it returns the output dict as the same type as the input dict
    so that the consuming serialization step knows the correct type info. The
    reverse function to this function is _detail_dictionary, which undoes
    the functionality of this function.

    Args:
        my_dict: A dictionary of python objects.

    Returns:
        Dict: A dictionary of the same type as the input of simplified
            objects.

    """
    pieces = list()
    # for dictionaries we want to simplify both the key and the value
    for key, value in my_dict.items():
        pieces.append((serde._simplify(key), serde._simplify(value)))

    return pieces


def _detail_dictionary(worker: AbstractWorker, my_dict: Dict) -> Dict:
    """
    This function is designed to operate in the opposite direction of
    _simplify_dictionary. It takes a dictionary of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_dict (Dict): a dictionary of simple python objects (including binary).

    Returns:
        tuple: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """
    pieces = {}
    # for dictionaries we want to detail both the key and the value
    for key, value in my_dict:
        detailed_key = serde._detail(worker, key)
        try:
            detailed_key = detailed_key.decode("utf-8")
        except AttributeError:
            pass

        detailed_value = serde._detail(worker, value)
        try:
            detailed_value = detailed_value.decode("utf-8")
        except AttributeError:
            pass

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
    }
)
