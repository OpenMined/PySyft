# future
from __future__ import annotations

import numpy as np
import pyarrow

from typing import Union
from numpy.typing import ArrayLike
from ...common.uid import UID
from .lazy_repeat_array import LazyRepeatArray
from .adp_tensor import Array, ADPTensor


class LazyPrimeArray(Array):

    def __init__(self, start, stop, shape):
        """ an array made of primes which is lazily evaluated

        start: the prime index this array begins with
        end: the prime inddex this array ends with
        shape: the shape of the array
        """
        self.start = start
        self.stop = stop
        self.shape = shape
        self._data_cache = None

    def simple_assets_for_serde(self):
        return [self.start, self.stop, self.shape]

    def reshape(self, *new_shape) -> LazyPrimeArray:
        if np.prod(self.shape) == np.prod(new_shape):

            result = LazyPrimeArray(start=self.start,
                                    stop=self.stop,
                                    shape=new_shape)
            return result
        else:
            raise Exception("New shape not compatible")

class GammaArray(ADPTensor):

    def __init__(self,
                 input2value,
                 input2minval,
                 input2maxval,
                 input2subjectprime,
                 shape,
                 is_linear,
                 input2scalarprime=None,
                 input2scalarprime_id=None,
                 value_cache=None,
                 minval_cache=None,
                 maxval_cache=None,
                 term=None,
                 coeff=None,
                 bias=None) -> None:

        # REPLACING SCALAR MANAGER ARE THE FOLLOWING NDARRAY LOOKUP TABLES
        self.input2value = input2value
        self.input2minval = input2minval
        self.input2maxval = input2maxval
        self.shape = shape

        if input2subjectprime.shape == self.input2value.shape:
            input2subjectprime = input2subjectprime.reshape(list(input2subjectprime.shape) + [1])

        # if an integer, it's assumed to be elementwise
        self.input2subjectprime = input2subjectprime

        # None == elementwise, unique primes for freshly created gammatensor, starting at 1
        self.input2scalarprime = input2scalarprime

        if input2scalarprime_id is None:
            # given no i2s id, ASSUME we're initializing this tensor for the first time!
            # which means all the caches are just copies of the data
            input2scalarprime_id = UID()
            value_cache = input2value
            minval_cache = input2minval
            maxval_cache = input2maxval
            is_linear = True

        self.input2scalarprime_id = input2scalarprime_id

        self.value_cache = value_cache
        self.minval_cache = minval_cache
        self.maxval_cache = maxval_cache

        self.is_linear = is_linear

        # tensor of polynomial terms - primes representing variables
        # None == elementwise, unique primes for freshly created gammatensor, starting at 1
        self._term = term

        # a tensor of coefficients - the floats which multiply by variables in polys
        # None == np.ones_like(term)
        self._coeff = coeff

        # a tensor of bias terms - scalars which are added to polys
        # None == np.zeros_like(term)
        self._bias = bias

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def term(self) -> Union[Array, ArrayLike]:
        if self._term is None:
            self._term = LazyPrimeArray(start=0, stop=np.prod(self.shape), shape=list(self.shape) + [1])
        return self._term

    @property
    def coeff(self) -> Union[Array, ArrayLike]:
        if self._coeff is None:
            self._coeff = LazyRepeatArray(data=1, shape=self.shape)
        return self._coeff

    @property
    def bias(self) -> Union[Array, ArrayLike]:
        if self._bias is None:
            self._bias = LazyRepeatArray(data=0, shape=self.shape)
        return self._bias

    def sum(self, axis=None) -> GammaArray:
        if axis is None:
            return GammaArray(input2value=self.input2value,
                           input2minval=self.input2minval,
                           input2maxval=self.input2maxval,
                           input2subjectprime=self.input2subjectprime,
                           shape=(),
                           is_linear=self.is_linear,
                           input2scalarprime=self.input2scalarprime,
                           input2scalarprime_id=self.input2scalarprime_id,
                           value_cache=self.value_cache.sum(),
                           minval_cache=self.minval_cache.sum(),
                           maxval_cache=self.maxval_cache.sum(),
                           term=self.term.reshape(1, self.size),
                           coeff=None if self._coeff is None else self.coeff.reshape(1, self.size),
                           bias=None if self._bias is None else self.bias.sum())

        else:
            raise Exception("Not sure how to run this yet")

    def deriv(self, inputs, input_mask=None):

        assert inputs.shape == self.input2value.shape

        # if someone doesn't pass in a mask we assume they
        # want to use all the inputs they're passing in
        if input_mask is None:
            input_mask = np.zeros_like(inputs)
        else:
            ""
            # if they do pass in a mask then 1s correspond
            # to data passed in and 0s to values from self.input2value

        assert inputs.shape == self.input2value.shape

        if self.is_linear:

            # TODO: lazyarray should know how to find the max coeff very
            # efficient instead of needing to hardcode this here
            if self._coeff is None:
                return np.zeros(self.shape)

        raise Exception("Ooops... can't compute max deriv of this yet...")

    def max_deriv(self, inputs, input_mask=None):

        assert inputs.shape == self.input2value.shape

        # if someone doesn't pass in a mask we assume they
        # want to use all the inputs they're passing in
        if input_mask is None:
            input_mask = np.zeros_like(inputs)
        else:
            ""
            # if they do pass in a mask then 1s correspond
            # to data passed in and 0s to values from self.input2value

        if self.is_linear:

            # TODO: lazyarray should know how to find the max coeff very
            # efficient instead of needing to hardcode this here
            if self._coeff is None:
                return np.zeros(self.shape)

        raise Exception("Ooops... can't compute max deriv of this yet...")

    #     def max_deriv_wrt_entity(self, entity_prime):

    def serialize(self):
        assets = list()
        assets.append(self.input2value)
        assets.append(self.input2minval.simple_assets_for_serde())
        assets.append(self.input2maxval.simple_assets_for_serde())
        assets.append(self.shape)
        assets.append(self.input2subjectprime)
        assets.append(self.input2scalarprime)
        #         assets.append(self.input2scalarprime_id)
        assets.append(self.value_cache)
        assets.append(self.minval_cache)
        assets.append(self.maxval_cache)
        assets.append(self._term.simple_assets_for_serde())
        assets.append(self._coeff)
        assets.append(self._bias)
        return pyarrow.serialize(assets).to_buffer()
