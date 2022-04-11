# future
from __future__ import annotations

# stdlib
from collections import deque
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple

# relative
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ...adp.data_subject_list import liststrtonumpyutf8
from ...adp.data_subject_list import numpyutf8tolist
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.serializable import serializable

if TYPE_CHECKING:
    # stdlib
    from dataclasses import dataclass
else:
    from flax.struct import dataclass

# third party
import flax
import jax
from jax import numpy as jnp
import numpy as np
from numpy.random import randint
from scipy.optimize import shgo

# relative
from ...adp.data_subject_ledger import DataSubjectLedger
from ...adp.data_subject_list import DataSubjectList
from ...adp.vectorized_publish import vectorized_publish


def create_lookup_tables(dictionary: dict) -> Tuple[List[str], dict, List[dict]]:
    index2key: List = [str(x) for x in dictionary.keys()]
    key2index: dict = {key: i for i, key in enumerate(index2key)}
    # Note this maps to GammaTensor, not to GammaTensor.value as name may imply
    index2values: List = [dictionary[i] for i in index2key]

    return index2key, key2index, index2values


def create_new_lookup_tables(
    dictionary: dict,
) -> Tuple[Deque[str], dict, Deque[dict], Deque[int]]:
    index2key: Deque = deque()
    key2index: dict = {}
    index2values: Deque = (
        deque()
    )  # Note this maps to GammaTensor, not to GammaTensor.value as name may imply
    index2size: Deque = deque()
    for index, key in enumerate(dictionary.keys()):
        key = str(key)
        index2key.append(key)
        key2index[key] = index
        index2values.append(dictionary[key])
        index2size.append(len(dictionary[key]))

    return index2key, key2index, index2values, index2size


def no_op(x: Dict[str, GammaTensor]) -> Dict[str, GammaTensor]:
    """A Private input will be initialized with this function.
    Whenever you manipulate a private input (i.e. add it to another private tensor),
    the result will have a different function. Thus we can check to see if the f
    """
    return x


def jax2numpy(value: jnp.array, dtype: np.dtype) -> np.array:
    # are we incurring copying here?
    return np.asarray(value, dtype=dtype)


def numpy2jax(value: np.array, dtype: np.dtype) -> jnp.array:
    return jnp.asarray(value, dtype=dtype)


@dataclass
@serializable(capnp_bytes=True)
class GammaTensor:
    value: jnp.array
    data_subjects: DataSubjectList
    min_val: float = flax.struct.field(pytree_node=False)
    max_val: float = flax.struct.field(pytree_node=False)
    is_linear: bool = True
    func: Callable = flax.struct.field(pytree_node=False, default_factory=lambda: no_op)
    id: str = flax.struct.field(
        pytree_node=False, default_factory=lambda: str(randint(0, 2**31 - 1))
    )  # TODO: Need to check if there are any scenarios where this is not secure
    inputs: jnp.array = np.array([], dtype=np.int64)
    state: dict = flax.struct.field(pytree_node=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.state) == 0:
            self.state[self.id] = self

    def run(self, state: dict) -> Callable:
        # we hit a private input
        if self.func is no_op:
            return self.func(state[self.id].value)
        return self.func(state)

    def __add__(self, other: Any) -> GammaTensor:
        state = dict()
        state.update(self.state)

        if isinstance(other, GammaTensor):

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other.run(state))

            state.update(other.state)
            value = self.value + other.value
            min_val = self.min_val + other.min_val
            max_val = self.max_val + other.max_val
        else:

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other)

            value = self.value + other
            min_val = self.min_val + other
            max_val = self.max_val + other

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_add,
            state=state,
        )

    def __mul__(self, other: Any) -> GammaTensor:
        state = dict()
        state.update(self.state)

        if isinstance(other, GammaTensor):

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other.run(state))

            state.update(other.state)
            value = self.value * other.value
        else:

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other)

            value = self.value * other

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=0,
            max_val=10,
            func=_mul,
            state=state,
        )

    def sum(self, *args: Tuple[Any, ...], **kwargs: Any) -> GammaTensor:
        def _sum(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sum(self.run(state))

        state = dict()
        state.update(self.state)

        value = jnp.sum(self.value)
        min_val = jnp.sum(self.min_val)
        max_val = jnp.sum(self.max_val)

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sum,
            state=state,
        )

    def sqrt(self) -> GammaTensor:
        def _sqrt(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sqrt(self.run(state))

        state = dict()
        state.update(self.state)

        value = jnp.sqrt(self.value)
        min_val = jnp.sqrt(self.min_val)
        max_val = jnp.sqrt(self.max_val)

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sqrt,
            state=state,
        )

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: Optional[float] = None,
        output_func: Callable = np.sum,
    ) -> jax.numpy.DeviceArray:
        # TODO: Add data scientist privacy budget as an input argument, and pass it
        # into vectorized_publish
        if sigma is None:
            sigma = self.value.mean() / 4

        return vectorized_publish(
            min_vals=self.min_val,
            max_vals=self.max_val,
            values=self.inputs,
            data_subjects=self.data_subjects,
            is_linear=self.is_linear,
            sigma=sigma,
            output_func=output_func,
            ledger=ledger,
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
        )

    def expand_dims(self, axis: int) -> GammaTensor:
        def _expand_dims(state: dict) -> jax.numpy.DeviceArray:
            return jnp.expand_dims(self.run(state), axis)

        state = dict()
        state.update(self.state)

        return GammaTensor(
            value=jnp.expand_dims(self.value, axis),
            data_subjects=self.data_subjects,
            min_val=self.min_val,
            max_val=self.max_val,
            func=_expand_dims,
            state=state,
        )

    def squeeze(self, axis: Optional[int] = None) -> GammaTensor:
        def _squeeze(state: dict) -> jax.numpy.DeviceArray:
            return jnp.squeeze(self.run(state), axis)

        state = dict()
        state.update(self.state)
        return GammaTensor(
            value=jnp.squeeze(self.value, axis),
            data_subjects=self.data_subjects,
            min_val=self.min_val,
            max_val=self.max_val,
            func=_squeeze,
            state=state,
        )

    def __len__(self) -> int:
        return len(self.value)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def lipschitz_bound(self) -> float:
        # TODO: Check if there are any functions for which lipschitz bounds shouldn't be computed
        # if dis(self.func) == dis(no_op):
        #     raise Exception

        print("Starting JAX JIT")
        fn = jax.jit(self.func)
        print("Traced self.func with jax's jit, now calculating gradient")
        grad_fn = jax.grad(fn)
        print("Obtained gradient, creating lookup tables")
        i2k, k2i, i2v, i2s = create_new_lookup_tables(self.state)

        print("created lookup tables, now getting bounds")
        i2minval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        i2maxval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        bounds = jnp.concatenate([i2minval, i2maxval], axis=1)
        print("Obtained bounds")
        # sample_input = i2minval.reshape(-1)
        _ = i2minval.reshape(-1)
        print("Obtained all inputs")

        def max_grad_fn(input_values: np.ndarray) -> float:
            vectors = {}
            n = 0
            for i, size_param in enumerate(i2s):
                vectors[i2k[i]] = input_values[n : n + size_param]  # noqa: E203
                n += size_param

            grad_pred = grad_fn(vectors)

            m = 0
            for value in grad_pred.values():
                m = max(m, jnp.max(value))

            # return negative because we want to maximize instead of minimize
            return -m

        print("starting SHGO")
        res = shgo(max_grad_fn, bounds, iters=1, constraints=tuple())
        print("Ran SHGO")
        # return negative because we flipped earlier
        return -float(res.fun)

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")

        gamma_tensor_struct: CapnpModule = schema.GammaTensor  # type: ignore
        gamma_msg = gamma_tensor_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        gamma_msg.magicHeader = serde_magic_header(type(self))

        # what is the difference between inputs and value which do we serde
        # do we need to serde func? if so how?
        # what about the state dict?

        gamma_msg.value = capnp_serialize(jax2numpy(self.value, dtype=self.value.dtype))
        gamma_msg.inputs = capnp_serialize(jax2numpy(self.inputs, self.inputs.dtype))
        gamma_msg.dataSubjectsIndexed = capnp_serialize(
            self.data_subjects.data_subjects_indexed
        )
        gamma_msg.oneHotLookup = capnp_serialize(
            liststrtonumpyutf8(self.data_subjects.one_hot_lookup)
        )
        gamma_msg.minVal = self.min_val
        gamma_msg.maxVal = self.max_val
        gamma_msg.isLinear = self.is_linear
        gamma_msg.id = self.id

        return gamma_msg.to_bytes_packed()

    @staticmethod
    def _bytes2object(buf: bytes) -> GammaTensor:
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")
        gamma_struct: CapnpModule = schema.GammaTensor  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # to pack or not to pack?
        # ndept_msg = ndept_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
        gamma_msg = gamma_struct.from_bytes_packed(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )

        value = capnp_deserialize(gamma_msg.value)
        inputs = capnp_deserialize(gamma_msg.inputs)
        data_subjects_indexed = capnp_deserialize(gamma_msg.dataSubjectsIndexed)
        one_hot_lookup = numpyutf8tolist(capnp_deserialize(gamma_msg.oneHotLookup))
        data_subjects = DataSubjectList(one_hot_lookup, data_subjects_indexed)
        min_val = gamma_msg.minVal
        max_val = gamma_msg.maxVal
        is_linear = gamma_msg.isLinear
        id_str = gamma_msg.id

        return GammaTensor(
            value=numpy2jax(value, dtype=value.dtype),
            data_subjects=data_subjects,
            min_val=min_val,
            max_val=max_val,
            is_linear=is_linear,
            inputs=numpy2jax(inputs, dtype=inputs.dtype),
            id=id_str,
        )
