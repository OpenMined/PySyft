# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
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

    def copy(self, order: Optional[str] = "K") -> DataSubjectList:
        return DataSubjectList(
            self.one_hot_lookup.copy(), self.data_subjects_indexed.copy(order=order)
        )

    def __len__(self) -> int:
        return len(self.data_subjects_indexed)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataSubjectList):
            if (self.data_subjects_indexed == other.data_subjects_indexed).all() and (  # type: ignore
                self.one_hot_lookup == other.one_hot_lookup
            ).all():
                return True
            return False
        return self == other

    def sum(self) -> DataSubjectList:
        # If sum is used without any arguments then the result is always a singular value
        return DataSubjectList(
            self.one_hot_lookup.copy(),
            self.data_subjects_indexed.flatten(),
        )

    @property
    def shape(self) -> Tuple:
        return self.data_subjects_indexed.shape


    @staticmethod
    def combine(
            dsl1: DataSubjectList, dsl2: DataSubjectList
    ) -> DataSubjectList:


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
        overlapping_indices = (overlap == True).nonzero()[0]  # noqa: E712
        # The DS at these indices are unique
        unique_indices = (overlap == False).nonzero()[0]  # noqa: E712

        # Suppressing E712 above because the recommended way (is True) does not work elementwise

        if len(overlapping_indices) == 0:  # If there's no overlap, our job is super simple
            return DataSubjectList(
                one_hot_lookup=np.concatenate((dsl1.one_hot_lookup, dsl2.one_hot_lookup)),
                data_subjects_indexed=np.stack(
                    (dsl1.data_subjects_indexed, dsl2.data_subjects_indexed + dsl1_uniques)
                ),
            )

        # Task 1- For the overlapping data subjects, we need to find the index already allotted to them.
        target_overlap_indices = np.searchsorted(
            bigger_list, search_terms.take(overlapping_indices)
        )

        # Now that we need to replace the previous indices with the new indices
        output_data_subjects_indexed = np.zeros_like(dsl2.data_subjects_indexed)
        for old_value, new_value in zip(overlapping_indices, target_overlap_indices):
            output_data_subjects_indexed[array_to_change == old_value] = new_value

        # Task 2- do the same but for unique data subjects
        unique_data_subjects = search_terms.take(unique_indices)

        output_one_hot_encoding = np.concatenate((bigger_list, unique_data_subjects))
        target_unique_indices = np.arange(len(bigger_list), len(output_one_hot_encoding))

        for old_value, new_value in zip(unique_indices, target_unique_indices):
            output_data_subjects_indexed[array_to_change == old_value] = new_value

        final_dsi = np.stack((unchanged_array, output_data_subjects_indexed))

        return DataSubjectList(
            one_hot_lookup=output_one_hot_encoding, data_subjects_indexed=final_dsi
        )
