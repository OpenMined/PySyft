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
from .entity import Entity


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
        if not isinstance(item, (Entity, str)):
            raise Exception(
                f"DataSubjectList entities must be List[Union[str, Entity]]. {type(item)}"
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

    # Temporarily remove as we are not using strings.
    # # one_hot_lookup is a numpy array of unicode strings which can't be serialized
    # __serde_overrides__ = {
    #     "one_hot_lookup": [liststrtonumpyutf8, numpyutf8tolist],
    # }

    def __init__(
        self,
        one_hot_lookup: np.ndarray[Union[Entity, str, np.integer]],
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

        # This will be the equivalent of the DataSubjectList.one_hot_indexed- a sorted array of all unique entities
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

    # def __getitem__(self, key: Union[int, slice, str]) -> Union[Entity, str]:
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
