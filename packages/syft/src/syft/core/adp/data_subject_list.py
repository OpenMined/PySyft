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


def get_output_shape(shape1, shape2):
    """
    When you insert values from one DP Tensor into another, this will help the DSL
    figure out what its output shape will be. (DSL.insert())

    Assumptions:
    - shape1 reflects the shape of the datapoints, i.e. (n, ... , r, c)
        - n = MAXIMUM number of data subjects for any given data point
        - r, c = rows/cols of data points
    - shape2 can either be:
        - the same shape as shape1 (if shape1 already has multiple DS per
            datapoint, or
        - 1 extra dimension that shape1 (if shape1 has only 1 DS per datapoint)
    """

    # if len(shape1) > len(shape2) or abs(len(shape2) - len(shape1)) > 1:
    #     raise NotImplementedError(f"Unsuitable shapes: {shape1}, {shape2}, {len(shape1)}, {len(shape2)}")

    first_dim = max(shape1[0], shape2[0])
    other_dims = shape1 if len(shape1) < len(shape2) else shape1[1:]
    output_size = (first_dim, *other_dims)
    return output_size


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
            ohl_comparison = (self.one_hot_lookup == other.one_hot_lookup)
            ohl_comparison = ohl_comparison if isinstance(ohl_comparison, bool) else ohl_comparison.all()
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

    def reshape(self, target_shape) -> DataSubjectList:
        return DataSubjectList(
            one_hot_lookup=self.one_hot_lookup,
            data_subjects_indexed=self.data_subjects_indexed.reshape(
                (-1, *target_shape)
            ),
        )

    @property
    def shape(self) -> Tuple:
        return self.data_subjects_indexed.shape

    @staticmethod
    def matmul(dsl1: DataSubjectList, dsl2: DataSubjectList):
        """
        Matmul can only be done when Tensor.child is a 2D array, thus Tensor.DSL is 2D (PhiTensor) or 3D (Gamma)
        Although Matmul first involves multiplication and then addition, implementing it with summation and then
        broadcasting was easier for DSL.
        """

        dsl1_target_shape = (*dsl1.shape[:-1], 1) if len(dsl1.shape) == 2 else (*dsl1.shape[1:-1], 1)
        dsl2_target_shape = (1, *dsl2.shape[:-3], dsl2.shape[-1]) if len(dsl2.shape) == 2 \
            else (*dsl2.shape[1:-2], 1, dsl2.shape[-1])

        summed_dsl1 = dsl1.sum(target_shape=dsl1_target_shape)
        summed_dsl2 = dsl2.sum(target_shape=dsl2_target_shape)

        # We need to project these data subject arrays to their entire row/column respectively
        dsl1_projection = np.ones((*summed_dsl1.shape[:-1], summed_dsl2.shape[-1]))  # *summed_dsl2.shape[1:-2],
        dsl2_projection = np.ones(
            (*summed_dsl2.shape[:-2], summed_dsl1.shape[-2], summed_dsl2.shape[-1]))  # *summed_dsl2.shape[1:-2],

        summed_dsl1.data_subjects_indexed = dsl1_projection * summed_dsl1.data_subjects_indexed
        summed_dsl2.data_subjects_indexed = dsl2_projection * summed_dsl2.data_subjects_indexed

        output_ds = DataSubjectList.combine_dsi(summed_dsl1, summed_dsl2)

        # This gets rid of redundant (repeating) DSL slices.
        output_ds.data_subjects_indexed = np.unique(output_ds.data_subjects_indexed, axis=0)
        return output_ds

    def dot(dsl1: DataSubjectList, dsl2: DataSubjectList):
        """
        a/b:
        0D/0D -> multiplication
        1D/1D -> multiplication
        1D/other -> sum product over last axis
        2D/2D -> matmul
        higher -> sum product over axis=-1 of a, axis=-2 of b

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        """

        dsl1_target_shape = (*dsl1.shape[1:-1], 1, 1)
        dsl2_target_shape = (1, 1, *dsl2.shape[1:-2], dsl2.shape[-1])

        summed_dsl1 = dsl1.sum(target_shape=dsl1_target_shape)
        summed_dsl2 = dsl2.sum(target_shape=dsl2_target_shape)

        # We need to project these data subject arrays to their entire row/column respectively
        dsl1_projection = np.ones((*summed_dsl1.shape[:-2], *summed_dsl2.shape[-2:]))
        dsl2_projection = np.ones(
            (summed_dsl2.shape[0], *summed_dsl1.shape[1:-2], *summed_dsl2.shape[-2:]))

        summed_dsl1.data_subjects_indexed = dsl1_projection * summed_dsl1.data_subjects_indexed
        summed_dsl2.data_subjects_indexed = dsl2_projection * summed_dsl2.data_subjects_indexed

        output_ds = DataSubjectList.combine_dsi(summed_dsl1, summed_dsl2)

        # This gets rid of redundant (repeating) DSL slices.
        output_ds.data_subjects_indexed = np.unique(output_ds.data_subjects_indexed, axis=0)
        return output_ds

    @staticmethod
    def index_dsl(tensor: Any, index):
        if tensor.shape == tensor.data_subjects.shape:
            return tensor.data_subjects[index]
        elif len(tensor.shape) < len(tensor.data_subjects.shape):
            return tensor.data_subjects[:, index]

    @staticmethod
    def combine_dsi(dsl1: DataSubjectList, dsl2: DataSubjectList):
        dsl1_dims = len(dsl1.shape)
        dsl2_dims = len(dsl2.shape)
        if dsl1_dims == dsl2_dims:
            output_shape = (dsl1.shape[0] + dsl2.shape[0], *dsl1.shape[1:])
        elif dsl1_dims - dsl2_dims == 1:
            dsl2.data_subjects_indexed = np.expand_dims(dsl2.data_subjects_indexed, 0)
            output_shape = (dsl1.shape[0] + dsl2.shape[0], *dsl1.shape[1:])
        elif dsl2_dims - dsl1_dims == 1:
            dsl1.data_subjects_indexed = np.expand_dims(dsl1.data_subjects_indexed, 0)
            output_shape = (dsl1.shape[0] + dsl2.shape[0], *dsl1.shape[1:])
        else:
            raise Exception(
                f"Shapes of DSLs incompatible- are they meant to be broadcasted: {dsl1.shape}, {dsl2.shape}")

        output_dsl = DataSubjectList(one_hot_lookup=np.array([]),
                                     data_subjects_indexed=np.empty(output_shape))

        output_dsl[:dsl1.shape[0]] = dsl1
        output_dsl[dsl1.shape[0]:] = dsl2
        return output_dsl

    def __getitem__(self, item) -> DataSubjectList:
        result = self.data_subjects_indexed[item]
        return DataSubjectList(
            one_hot_lookup=self.one_hot_lookup,  # np.unique(self.one_hot_lookup[result]),
            data_subjects_indexed=result,
        )

    def __setitem__(self, key, value) -> None:
        if isinstance(value, DataSubjectList):
            overlapping_ds = np.isin(value.one_hot_lookup, self.one_hot_lookup)
            copied = value.copy()
            if any(overlapping_ds):
                # Find which Data Subjects have been assigned an index already
                search = overlapping_ds == True  # noqa: E712
                overlapping_indices = search.nonzero()[0]

                # Find what index they've been assigned
                target_overlap_indices = np.searchsorted(
                    self.one_hot_lookup, copied.one_hot_lookup.take(overlapping_indices)
                )

                # Give them the index that was assigned to them
                for old_value, new_value in zip(
                    overlapping_indices, target_overlap_indices
                ):
                    copied.data_subjects_indexed[
                        copied.data_subjects_indexed == old_value
                    ] = new_value

                # Now do the same but for unique data subjects
                unique_indices = np.invert(search).nonzero()[0]  # noqa: E712
                unique_data_subjects = copied.data_subjects_indexed.take(unique_indices)

                output_one_hot_encoding = np.append(
                    self.one_hot_lookup, unique_data_subjects
                )
                target_unique_indices = np.arange(
                    len(self.data_subjects_indexed), len(output_one_hot_encoding)
                )

                for old_value, new_value in zip(unique_indices, target_unique_indices):
                    copied.data_subjects_indexed[
                        copied.data_subjects_indexed == old_value
                    ] = new_value

                self.one_hot_lookup = output_one_hot_encoding
                self.data_subjects_indexed[key] = copied.data_subjects_indexed

            else:
                copied.data_subjects_indexed += len(self.one_hot_lookup)
                output_lookup = np.append(self.one_hot_lookup, copied.one_hot_lookup)
                new_data_subjects = self.data_subjects_indexed.copy()
                new_data_subjects[key] = copied.data_subjects_indexed

                self.one_hot_lookup = output_lookup
                self.data_subjects_indexed = new_data_subjects
        else:
            raise NotImplementedError(f"Undefined behaviour for type: {type(value)}")

    @staticmethod
    def insert(dsl1: DataSubjectList, dsl2: DataSubjectList, index) -> DataSubjectList:
        output_shape = get_output_shape(dsl1.shape, dsl2.shape)

        # TODO: Broadcasting with [1] over multiple indices?
        dsl1_len = dsl1.shape[0] if len(dsl1.shape) >= len(dsl2.shape) else 1
        dsl2_len = dsl2.shape[0]

        if dsl1.shape == dsl2.shape:
            # This is not an insertion, this is a complete replacement.
            return dsl2.copy()
        elif dsl1_len < dsl2_len:
            # We need to create a bigger array
            bigger_array = np.full(output_shape, np.nan)

            # Fill current dsl1 values into this bigger_array
            # TODO: Ask Madhava if there's a smarter/vectorized way to do this
            if dsl1.shape[0] == 1:
                bigger_array[0] = dsl1.data_subjects_indexed
            else:
                for i in range(dsl1.shape[0]):
                    bigger_array[i] = dsl1.data_subjects_indexed[i]

            output_dsl = DataSubjectList(one_hot_lookup=dsl1.one_hot_lookup, data_subjects_indexed=bigger_array)
            dsl2_copy = dsl2.copy()
            output_dsl[:, index] = dsl2_copy
            return output_dsl

        elif dsl1_len == dsl2_len:
            # it fits perfectly
            output_dsl = dsl1.copy()
            output_dsl[:, index] = dsl2
            return output_dsl
        else:
            # add some nans
            extra_nans = np.array([np.nan] * (dsl1_len - dsl2_len) * np.prod(dsl2.shape[1:])).reshape(
                (dsl1_len - dsl2_len, *dsl2.shape[1:]))
            array_to_append = np.concatenate((dsl2.data_subjects_indexed, extra_nans)).squeeze()
            new_dsl = dsl2.copy()
            new_dsl.data_subjects_indexed = array_to_append

            output_dsl = dsl1.copy()
            output_dsl[:, index] = new_dsl
            return output_dsl

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
        # TODO: Check if DSLs need to be flattened before appending, if different sizes?

        """
        Unlike Combine() which creates a DSL for a GammaTensor where one data point is owned
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
