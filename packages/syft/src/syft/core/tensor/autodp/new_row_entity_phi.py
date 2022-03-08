# future
from __future__ import annotations

import pyarrow
import numpy as np
from numpy.typing import ArrayLike
from typing import Union

from .new_gamma_array import GammaArray
from .single_entity_phi import SingleEntityPhiTensor
from .lazy_repeat_array import LazyRepeatArray
from .adp_tensor import ADPTensor, Array


class NewRowEntityPhiArray(ADPTensor):

    def __init__(self,
                 rows,
                 min_vals: Union[Array, ArrayLike],
                 max_vals: Union[Array, ArrayLike],
                 data_subjects,
                 row_type=SingleEntityPhiTensor):

        self.rows = rows
        self.minv = min_vals
        self.maxv = max_vals
        self.subs = data_subjects
        self.row_type = row_type

    def sum(self, axis=None):
        return self.gamma.sum(axis=axis)

    def serialize(self):
        # TODO: this is only valid if minv and maxv are LazyRepeatArrays
        assets = [self.rows,
                  self.minv.simple_assets_for_serde(),
                  self.maxv.simple_assets_for_serde(),
                  self.subs,
                  self.row_type]
        return pyarrow.serialize(assets).to_buffer()

    @staticmethod
    def deserialize(blob):
        assets = pyarrow.deserialize(blob)
        rows = assets[0]
        minv = LazyRepeatArray.deserialize_from_simple_assets(assets[1])
        maxv = LazyRepeatArray.deserialize_from_simple_assets(assets[2])
        subs = assets[3]
        row_type = assets[4]
        return NewRowEntityPhiArray(rows=rows,
                        min_vals=minv,
                        max_vals=maxv,
                        data_subjects=subs,
                        row_type=row_type)

    @property
    def shape(self):
        return self.rows.shape

    @property
    def gamma(self) -> GammaArray:
        if self.row_type == SingleEntityPhiTensor:
            return GammaArray(input2value=self.rows,
                           input2minval=self.minv,
                           input2maxval=self.maxv,
                           input2subjectprime=self.subs,
                           shape=self.shape,
                           is_linear=True)
        else:
            raise Exception("Sorry don't know how to convert this to gamma yet.")


