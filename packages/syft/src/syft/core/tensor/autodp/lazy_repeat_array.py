# future
from __future__ import annotations

from ...common.serde.serializable import serializable
import numpy as np
from .adp_tensor import Array


@serializable()
class LazyRepeatArray(Array):
    """when data is repeated along one or more dimensions, store it using this lazyrepeatarray so that
    you can save on RAM and CPU when computing with it."""

    def __init__(self, data, shape):
        """
        data: the raw data values without repeats
        shape: the shape of 'data' if repeats were included
        """

        # NOTE: all additional arguments are assumed to be broadcast if dims are shorter
        # than that of data. Example: if data.shape == (2,3,4) and min_vals.shape == (2,3),
        # then it's assumed that the full min_vals.shape is actually (2,3,4) where the last
        # dim is simply copied. Example2: if data.shape == (2,3,4) and min_vals.shape == (2,1,4),
        #  then the middle dimension is supposed to be copied to be min_vals.shape == (2,3,4) if
        # necessary. This is just to keep the memory footprint (and computation) as small as
        # possible.

        if isinstance(data, (bool, int, float)):
            data = np.array(data)

        self.data = data
        self.shape = shape

    def __sub__(self, other: LazyRepeatArray) -> LazyRepeatArray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if self.shape != other.shape:
            raise Exception("cannot subtract tensors with different shapes")

        if self.data.shape == other.data.shape:
            return LazyRepeatArray(data=self.data - other.data, shape=self.shape)

        raise Exception("not sure how to do this yet")

    def __mul__(self, other: LazyRepeatArray) -> LazyRepeatArray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if self.shape != other.shape:
            raise Exception("cannot subtract tensors with different shapes")

        if self.data.shape == other.data.shape:
            return LazyRepeatArray(data=self.data * other.data, shape=self.shape)

        raise Exception("not sure how to do this yet")

    def __pow__(self, exponent) -> LazyRepeatArray:
        if exponent == 2:
            return self * self
        raise Exception("not sure how to do this yet")

    def simple_assets_for_serde(self):
        return [self.data, self.shape]

    @staticmethod
    def deserialize_from_simple_assets(assets):
        return LazyRepeatArray(data=assets[0],
                               shape=assets[1])

    @property
    def size(self):
        return np.prod(self.shape)

    def sum(self, axis=None):
        if axis is None:
            if self.data.size == 1:
                return np.array(self.data * self.size).flatten()
            else:
                raise Exception("not sure how to do this yet")
        else:
            raise Exception("not sure how to do this yet")