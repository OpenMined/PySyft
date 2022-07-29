# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Set
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

# relative
from ..common.serde.serializable import serializable
from .data_subject import DataSubject


# allow us to serialize and deserialize np.arrays with strings inside as two np.arrays
# one containing the uint8 bytes and the other the offsets between strings
# TODO: Should move to a vectorized version.
def numpyutf8tolist(string_index: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    index_length = int(string_index[-1])
    index_array = string_index[-(index_length + 1) : -1]  # noqa
    string_array: np.ndarray = string_index[: -(index_length + 1)]
    output_bytes: bytes = string_array.astype(np.uint8).tobytes()
    output_list = []
    last_offset = 0
    for offset in index_array:
        chars = output_bytes[last_offset:offset]
        final_string = chars.decode("utf-8")
        last_offset = offset
        output_list.append(final_string)
    return np.array(output_list)


def liststrtonumpyutf8(string_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bytes_list = []
    indexes = []
    offset = 0
    for item in string_list:
        if not isinstance(item, (DataSubject, str)):
            raise Exception(
                f"DataSubjectList entities must be List[Union[str, DataSubject]]. {type(item)}"
            )
        name = item if isinstance(item, str) else item.name
        name_bytes = name.encode("utf-8")
        offset += len(name_bytes)
        indexes.append(offset)
        bytes_list.append(name_bytes)

    np_bytes = np.frombuffer(b"".join(bytes_list), dtype=np.uint8)
    np_bytes = np_bytes.astype(np.uint64)
    np_indexes = np.array(indexes, dtype=np.uint64)
    index_length = np.array([len(np_indexes)], dtype=np.uint64)
    output_array = np.concatenate([np_bytes, np_indexes, index_length])
    return output_array


@serializable(recursive_serde=True)
class DataSubjectList:
    __attr_allowlist__ = ("one_hot_lookup", "data_subjects_indexed")
    __slots__ = ("one_hot_lookup", "data_subjects_indexed")

    # one_hot_lookup is a numpy array of unicode strings which can't be serialized
    __serde_overrides__ = {
        "one_hot_lookup": [liststrtonumpyutf8, numpyutf8tolist],
    }

    def __init__(
        self,
        one_hot_lookup: np.ndarray[Union[DataSubject, str, np.integer]],
        data_subjects_indexed: np.ndaray,
    ) -> None:
        self.one_hot_lookup = one_hot_lookup
        self.data_subjects_indexed = data_subjects_indexed

    @staticmethod
    def from_series(entities_dataframe_slice: pd.Series) -> DataSubjectList:
        """Given a Pandas Series object (such as from
        getting a column from a pandas DataFrame, return an DataSubjectList"""

        # This will be the equivalent of the DataSubjectList.data_subjects_indexed
        if not isinstance(entities_dataframe_slice, np.ndarray):
            data_subjects = entities_dataframe_slice.to_numpy()
        else:
            data_subjects = entities_dataframe_slice

        # data_subjects = (
        #     entities_dataframe_slice.to_numpy()
        #     if not isinstance(entities_dataframe_slice, np.ndarray)
        #     else entities_dataframe_slice
        # )
        unique_data_subjects, data_subjects_indexed = np.unique(
            data_subjects, return_inverse=True
        )

        # This will be the equivalent of the DataSubjectList.one_hot_indexed- a sorted array of all unique data_subjects
        unique_data_subjects = unique_data_subjects.astype(np.str_)

        return DataSubjectList(
            one_hot_lookup=unique_data_subjects,
            data_subjects_indexed=data_subjects_indexed,
        )

    @staticmethod
    def from_objs(entities: Union[np.ndarray, list]) -> DataSubjectList:

        entities = np.array(entities, copy=False)
        one_hot_lookup, entities_indexed = np.unique(entities, return_inverse=True)
        if entities_indexed.shape != entities.shape:
            entities_indexed.resize(entities.shape)
        return DataSubjectList(one_hot_lookup, entities_indexed)

    # def __getitem__(self, key: Union[int, slice, str]) -> Union[DataSubject, str]:
    #     return self.one_hot_lookup[self.data_subjects_indexed[key]]

    def copy(self) -> DataSubjectList:
        return DataSubjectList(
            self.one_hot_lookup.copy(), self.data_subjects_indexed.copy()
        )

    def __len__(self) -> int:
        return len(self.data_subjects_indexed)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataSubjectList):
            dsi_comparison = self.data_subjects_indexed == other.data_subjects_indexed
            ohl_comparison = self.one_hot_lookup == other.one_hot_lookup
            ohl_comparison = (
                ohl_comparison
                if isinstance(ohl_comparison, bool)
                else ohl_comparison.all()
            )
            if isinstance(dsi_comparison, bool):
                if dsi_comparison is True and ohl_comparison is True:
                    return True
                else:
                    return False
            elif isinstance(dsi_comparison, np.ndarray):
                if dsi_comparison.all() is True and ohl_comparison is True:
                    return True
                else:
                    return False
        return self == other

    def sum(self, target_shape: tuple) -> DataSubjectList:
        """
        ::target_shape:: is the shape that the data (DP Tensor.child) has.
        """
        return DataSubjectList(
            self.one_hot_lookup.copy(),
            self.data_subjects_indexed.reshape((-1, *target_shape)),
        )

    def flatten(self) -> DataSubjectList:
        return DataSubjectList(
            one_hot_lookup=self.one_hot_lookup.copy(),
            data_subjects_indexed=self.data_subjects_indexed.flatten(),
        )

    def transpose(self: DataSubjectList, axes: Tuple) -> DataSubjectList:
        return DataSubjectList(
            self.one_hot_lookup, np.transpose(self.data_subjects_indexed, axes=axes)
        )

    @staticmethod
    def combine(dsl1: DataSubjectList, dsl2: DataSubjectList) -> DataSubjectList:

        """
        From Ishan's PR: https://github.com/OpenMined/PySyft/pull/6490/

        This code lets us combine two Data Subject Lists.
        This is essential when going from PhiTensors -> GammaTensors.

        The benefit of this code over recalling np.unique(return_indexed) is that we don't
        iterate over the entire 2nd data subject list.

        :param dsl1:
        :param dsl2:
        :return:
        """

        if dsl1.one_hot_lookup.size == 0:
            return dsl2
        elif dsl2.one_hot_lookup.size == 0:
            return dsl1
        else:

            # dsl1_uniques = dsl1.num_uniques
            # dsl2_uniques = dsl2.num_uniques
            dsl1_uniques = len(dsl1.one_hot_lookup)
            dsl2_uniques = len(dsl2.one_hot_lookup)

            if dsl1_uniques >= dsl2_uniques:
                search_terms = dsl2.one_hot_lookup
                bigger_list = dsl1.one_hot_lookup
                array_to_change = dsl2.data_subjects_indexed
                unchanged_array = dsl1.data_subjects_indexed
            else:
                search_terms = dsl1.one_hot_lookup
                bigger_list = dsl2.one_hot_lookup
                array_to_change = dsl1.data_subjects_indexed
                unchanged_array = dsl2.data_subjects_indexed

            # Array of True/False depending on whether search term exists in bigger list
            overlap = np.isin(
                search_terms, bigger_list
            )  # TODO: is there a way to use np.searchsorted w/o the index 0 problem?

            # The DS at these indices of search_terms exist in bigger_list
            search = overlap == True  # noqa: E712
            overlapping_indices = search.nonzero()[0]
            # The DS at these indices are unique
            unique_indices = np.invert(search).nonzero()[0]  # noqa: E712

            # Suppressing E712 above because the recommended way (is True) does not work elementwise

            if (
                len(overlapping_indices) == 0
            ):  # If there's no overlap, our job is super simple
                return DataSubjectList(
                    one_hot_lookup=np.concatenate(
                        (dsl1.one_hot_lookup, dsl2.one_hot_lookup)
                    ),
                    data_subjects_indexed=np.stack(
                        (
                            dsl1.data_subjects_indexed,
                            dsl2.data_subjects_indexed + dsl1_uniques,
                        )
                    ),
                )

            # Task 1- For the overlapping data subjects, we need to find the index already allotted to them.
            target_overlap_indices = np.searchsorted(
                bigger_list, search_terms.take(overlapping_indices)
            )

            # Now that we need to replace the previous indices with the new indices
            output_data_subjects_indexed = np.zeros_like(dsl2.data_subjects_indexed)
            for old_value, new_value in zip(
                overlapping_indices, target_overlap_indices
            ):
                output_data_subjects_indexed[array_to_change == old_value] = new_value

            # Task 2- do the same but for unique data subjects
            unique_data_subjects = search_terms.take(unique_indices)

            output_one_hot_encoding = np.concatenate(
                (bigger_list, unique_data_subjects)
            )
            target_unique_indices = np.arange(
                len(bigger_list), len(output_one_hot_encoding)
            )

            for old_value, new_value in zip(unique_indices, target_unique_indices):
                output_data_subjects_indexed[array_to_change == old_value] = new_value

            final_dsi = np.stack((unchanged_array, output_data_subjects_indexed))

            return DataSubjectList(
                one_hot_lookup=output_one_hot_encoding, data_subjects_indexed=final_dsi
            )

    @staticmethod
    def absorb(dsl1: DataSubjectList, dsl2: DataSubjectList) -> DataSubjectList:
        # TODO: Check if DataSubjectArrays need to be flattened before appending, if different sizes?

        """
        Unlike Combine() which creates a DataSubjectArray for a GammaTensor where one data point is owned
        by multiple data subjects, Absorb() creates a GammaTensor where each data point is
        still only owned by a single data subject.


        :param dsl1:
        :param dsl2:
        :return:
        """

        if dsl1.one_hot_lookup.size == 0:
            return dsl2
        elif dsl2.one_hot_lookup.size == 0:
            return dsl1
        else:

            # dsl1_uniques = dsl1.num_uniques
            # dsl2_uniques = dsl2.num_uniques
            dsl1_uniques = len(dsl1.one_hot_lookup)
            dsl2_uniques = len(dsl2.one_hot_lookup)

            if dsl1_uniques >= dsl2_uniques:
                search_terms = dsl2.one_hot_lookup
                bigger_list = dsl1.one_hot_lookup
                array_to_change = dsl2.data_subjects_indexed
                unchanged_array = dsl1.data_subjects_indexed
            else:
                search_terms = dsl1.one_hot_lookup
                bigger_list = dsl2.one_hot_lookup
                array_to_change = dsl1.data_subjects_indexed
                unchanged_array = dsl2.data_subjects_indexed

            # Array of True/False depending on whether search term exists in bigger list
            overlap = np.isin(
                search_terms, bigger_list
            )  # TODO: is there a way to use np.searchsorted w/o the index 0 problem?

            # The DS at these indices of search_terms exist in bigger_list
            search = overlap == True  # noqa: E712
            overlapping_indices = search.nonzero()[0]
            # The DS at these indices are unique
            unique_indices = np.invert(search).nonzero()[0]

            # Suppressing E712 above because the recommended way (is True) does not work elementwise

            if (
                len(overlapping_indices) == 0
            ):  # If there's no overlap, our job is super simple
                return DataSubjectList(
                    one_hot_lookup=np.concatenate(
                        (dsl1.one_hot_lookup, dsl2.one_hot_lookup)
                    ),
                    data_subjects_indexed=np.append(  # Should this be concatenate?
                        dsl1.data_subjects_indexed,
                        dsl2.data_subjects_indexed + dsl1_uniques,
                    ),
                )

            # Task 1- For the overlapping data subjects, we need to find the index already allotted to them.
            target_overlap_indices = np.searchsorted(
                bigger_list, search_terms.take(overlapping_indices)
            )

            # Now that we need to replace the previous indices with the new indices
            output_data_subjects_indexed = np.zeros_like(dsl2.data_subjects_indexed)
            for old_value, new_value in zip(
                overlapping_indices, target_overlap_indices
            ):
                output_data_subjects_indexed[array_to_change == old_value] = new_value

            # Task 2- do the same but for unique data subjects
            unique_data_subjects = search_terms.take(unique_indices)

            output_one_hot_encoding = np.concatenate(
                (bigger_list, unique_data_subjects)
            )
            target_unique_indices = np.arange(
                len(bigger_list), len(output_one_hot_encoding)
            )

            for old_value, new_value in zip(unique_indices, target_unique_indices):
                output_data_subjects_indexed[array_to_change == old_value] = new_value

            final_dsi = np.append(unchanged_array, output_data_subjects_indexed)

            return DataSubjectList(
                one_hot_lookup=output_one_hot_encoding, data_subjects_indexed=final_dsi
            )


def numpyutf8todslarray(input_index: Tuple[np.ndarray, np.ndarray]) -> ArrayLike:
    """Decodes utf-8 encoded numpy array to DataSubjectArray.

    Args:
        string_index (Tuple[np.ndarray, np.ndarray]): encoded array

    Returns:
        np.ndarray: decoded DataSubjectArray.
    """
    shape_length = int(input_index[-1])
    shape = tuple(input_index[-(shape_length + 1) : -1])  # noqa
    string_index = input_index[: -(shape_length + 1)]
    index_length = int(string_index[-1])
    index_array = string_index[-(index_length + 1) : -1]  # noqa
    string_array: np.ndarray = string_index[: -(index_length + 1)]
    output_bytes: bytes = string_array.astype(np.uint8).tobytes()
    output_list = []
    last_offset = 0
    for offset in index_array:
        chars = output_bytes[last_offset:offset]
        final_string = DataSubjectArray.fromstring(chars.decode("utf-8"))
        last_offset = offset
        output_list.append(final_string)
    return np.array(output_list).reshape(shape)


def dslarraytonumpyutf8(string_list: np.ndarray) -> ArrayLike:
    """Encodes DataSubjectArray to utf-8 encoded numpy array.

    Args:
        string_list (np.ndarray): DataSubjectArray to be encoded

    Raises:
        Exception: DataSubject is not a DataSubjectArray

    Returns:
        Tuple[np.ndarray, np.ndarray]: utf-8 encoded int Numpy array
    """
    # print("dsl list before", string_list)
    array_shape = string_list.shape
    string_list = string_list.flatten()
    bytes_list = []
    indexes = []
    offset = 0
    # print("dsl list ", string_list)
    for item in string_list:
        if not isinstance(item, DataSubjectArray):
            raise Exception(
                f"DataSubjectList entities must be  DataSubject. {type(item)}"
            )
        name = item.tostring()
        name_bytes = name.encode("utf-8")
        offset += len(name_bytes)
        indexes.append(offset)
        bytes_list.append(name_bytes)

    np_bytes = np.frombuffer(b"".join(bytes_list), dtype=np.uint8)
    np_bytes = np_bytes.astype(np.uint64)
    np_indexes = np.array(indexes, dtype=np.uint64)
    index_length = np.array([len(np_indexes)], dtype=np.uint64)
    shape = np.array(array_shape, dtype=np.uint64)
    shape_length = np.array([len(shape)], dtype=np.uint64)
    output_array = np.concatenate(
        [np_bytes, np_indexes, index_length, shape, shape_length]
    )

    return output_array


@serializable(recursive_serde=True)
class DataSubjectArray:
    __attr_allowlist__ = ("data_subjects",)

    delimiter = ","

    def __init__(self, data_subjects: Union[str, List[str], Set[str]]):
        self.data_subjects = set(data_subjects)

    def __len__(self) -> int:
        return len(self.data_subjects)

    def tostring(self) -> str:
        return f"{self.delimiter}".join(self.data_subjects)

    @classmethod
    def fromstring(cls, input_string: str) -> DataSubjectArray:
        return DataSubjectArray(set(input_string.split(f"{cls.delimiter}")))

    def __add__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __sub__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __mul__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __ge__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __le__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __gt__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def exp(self) -> DataSubjectArray:
        return DataSubjectArray(self.data_subjects)

    def __lt__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __truediv__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __rmatmul__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(self.data_subjects.union(other.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __rtruediv__(self, other: Union[DataSubjectArray, Any]) -> DataSubjectArray:
        if isinstance(other, DataSubjectArray):
            return DataSubjectArray(other.data_subjects.union(self.data_subjects))
        else:
            return DataSubjectArray(self.data_subjects)

    def __repr__(self) -> str:
        return "DataSubjectArray: " + str(self.data_subjects.__repr__())

    def __iter__(self) -> Iterator[Any]:
        for data_subject in self.data_subjects:
            yield data_subject

    def __contains__(self, item: Union[str, DataSubjectArray]) -> bool:
        if isinstance(item, DataSubjectArray):
            return self.data_subjects.isdisjoint(item.data_subjects)
        else:
            return self.data_subjects.isdisjoint(set(item))

    def conjugate(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> DataSubjectArray:
        return DataSubjectArray(self.data_subjects)

    def subtract(
        self,
        x: Union[DataSubjectArray, Any],
        y: Union[DataSubjectArray, Any],
        *args: List[Any],
        **kwargs: Dict[Any, Any],
    ) -> DataSubjectArray:
        if isinstance(y, DataSubjectArray) and isinstance(x, DataSubjectArray):
            return DataSubjectArray(x.data_subjects.union(y.data_subjects))
        elif isinstance(y, DataSubjectArray):
            return DataSubjectArray(y.data_subjects)
        elif isinstance(x, DataSubjectArray):
            return DataSubjectArray(x.data_subjects)
        else:
            raise ValueError(
                f"Either X:{type(x)} Y:{type(y)} should be DataSubjectArray"
            )

    def multiply(
        self,
        x: Union[DataSubjectArray, Any],
        y: Union[DataSubjectArray, Any],
        *args: List[Any],
        **kwargs: Dict[Any, Any],
    ) -> DataSubjectArray:
        if isinstance(y, DataSubjectArray) and isinstance(x, DataSubjectArray):
            return DataSubjectArray(x.data_subjects.union(y.data_subjects))
        elif isinstance(y, DataSubjectArray):
            return DataSubjectArray(y.data_subjects)
        elif isinstance(x, DataSubjectArray):
            return DataSubjectArray(x.data_subjects)
        else:
            raise ValueError(
                f"Either X:{type(x)} Y:{type(y)} should be DataSubjectArray"
            )

    def __eq__(self, other: Union[DataSubjectArray, Any]) -> bool:
        if isinstance(other, DataSubjectArray):
            return self.data_subjects == other.data_subjects
        elif isinstance(other, np.ndarray):
            return np.array(self) == other
        else:
            raise NotImplementedError

    def log(self) -> DataSubjectArray:
        return DataSubjectArray(self.data_subjects)

    def real(self) -> DataSubjectArray:
        return DataSubjectArray(self.data_subjects)

    def var(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> DataSubjectArray:
        return (self - np.mean(self)) * (self - np.mean(self))

    def sqrt(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> DataSubjectArray:
        return DataSubjectArray(self.data_subjects)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> ArrayLike:  # type: ignore
        method_name = ufunc.__name__
        print("method_name", method_name)
        method = getattr(self, method_name, None)
        if method is not None:
            return method(*inputs, **kwargs)
        else:
            raise NotImplementedError(
                f"Method: {method_name} not implemented in DataSubjectArray"
            )

    @staticmethod
    def from_objs(input_subjects: Union[np.ndarray, list]) -> ArrayLike:
        # TODO: When the user passes the data subjects they might pass it as list
        # specifying the entity per row, but in our new notation we want it to be
        # per data point, we should make sure that we implement in such a way we expand
        # the datasubjects automatically for row to data point mapping.
        if not isinstance(input_subjects, np.ndarray):
            input_subjects = np.array(input_subjects)

        data_map = (
            lambda x: DataSubjectArray([str(x)])
            if not isinstance(x, DataSubjectArray)
            else x
        )
        map_function = np.vectorize(data_map)

        return map_function(input_subjects)
