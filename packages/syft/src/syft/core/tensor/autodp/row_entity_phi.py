# future
from __future__ import annotations

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft relative
from ....core.common.serde.serializable import Serializable
from ....lib.util import full_name_with_name
from ....proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import bind_protobuf
from ...common.serde.serialize import _serialize as serialize
from ..passthrough import PassthroughTensor
from ..passthrough import implements
from ..passthrough import is_acceptable_simple_type
from .initial_gamma import InitialGammaTensor
from .single_entity_phi import SingleEntityPhiTensor


@bind_protobuf
class RowEntityPhiTensor(PassthroughTensor, Serializable):
    def __init__(self, rows, check_shape=True):
        super().__init__(rows)

        if check_shape:
            shape = rows[0].shape
            for row in rows[1:]:
                if shape != row.shape:
                    raise Exception(
                        f"All rows in RowEntityPhiTensor must match: {shape} != {row.shape}"
                    )

    @property
    def scalar_manager(self):
        return self.child[0].scalar_manager

    @property
    def min_vals(self):
        return np.concatenate([x.min_vals for x in self.child]).reshape(self.shape)

    @property
    def max_vals(self):
        return np.concatenate([x.max_vals for x in self.child]).reshape(self.shape)

    @property
    def value(self):
        return np.concatenate([x.child for x in self.child]).reshape(self.shape)

    @property
    def entities(self):
        return np.array(
            [[x.entity] * np.array(x.shape).prod() for x in self.child]
        ).reshape(self.shape)

    @property
    def gamma(self):
        return self.create_gamma()

    def create_gamma(self, scalar_manager=None):

        if scalar_manager is None:
            scalar_manager = self.scalar_manager

        return InitialGammaTensor(
            values=self.value,  # 5 x 10 data
            min_vals=self.min_vals,  # 5 x 10 minimum values
            max_vals=self.max_vals,  # 5 x 10 maximum values
            entities=self.entities,  # list of 5 entities
            scalar_manager=scalar_manager,
        )

    @property
    def shape(self):
        return [len(self.child)] + list(self.child[0].shape)

    def __add__(self, other):

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] + other)
                else:
                    new_list.append(self.child[i] + other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __add__: {len(self.child)} != {len(other.child)}"
            )

    def __sub__(self, other):
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] - other)
                else:
                    new_list.append(self.child[i] - other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __sub__: {len(self.child)} != {len(other.child)}"
            )

    def __mul__(self, other):

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    if isinstance(other, (int, bool, float)):
                        new_list.append(self.child[i] * other)
                    else:
                        new_list.append(self.child[i] * other[i])
                else:
                    new_list.append(self.child[i] * other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __mul__: {len(self.child)} != {len(other.child)}"
            )

    def __truediv__(self, other):

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] / other)
                else:
                    new_list.append(self.child[i] / other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __truediv__: {len(self.child)} != {len(other.child)}"
            )

    def repeat(self, repeats, axis=None):

        if axis is None:
            raise Exception(
                "Conservatively, RowEntityPhiTensor doesn't yet support repeat(axis=None)"
            )

        if axis == 0 or axis == -len(self.shape):
            new_list = list()
            for r in range(repeats):
                for row in self.child:
                    new_list.append(row)
            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        elif axis > 0:
            new_list = list()
            for row in self.child:
                new_list.append(row.repeat(repeats, axis=axis - 1))
            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        # axis is negative
        elif abs(axis) < len(self.shape):
            new_list = list()
            for row in self.child:
                new_list.append(row.repeat(repeats, axis=axis))
            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                "'axis' arg is negative and strangely large... not sure what to do."
            )

    def reshape(self, *shape):

        if shape[0] != self.shape[0]:
            raise Exception(
                "For now, you can't reshape the first dimension because that would"
                + "probably require creating a gamma tensor."
            )

        new_list = list()
        for row in self.child:
            new_list.append(row.reshape(shape[1:]))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def sum(self, *args, axis=None, **kwargs):

        if axis is None or axis == 0:
            return self.gamma.sum(axis=axis)

        new_list = list()
        for row in self.child:
            new_list.append(row.sum(*args, axis=axis - 1, **kwargs))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def transpose(self, *dims):
        print(dims)
        if dims[0] != 0:
            raise Exception("Can't move dim 0 in RowEntityPhiTensor at this time")

        new_dims = list(np.array(dims[1:]))

        new_list = list()
        for row in self.child:
            new_list.append(row.transpose(*new_dims))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def _object2proto(self) -> Tensor_PB:
        arrays = []
        tensors = []
        if len(self.child) > 0 and isinstance(self.child[0], np.ndarray):
            use_tensors = False
            arrays = [serialize(child) for child in self.child]
        else:
            use_tensors = True
            tensors = [serialize(child) for child in self.child]

        return Tensor_PB(
            obj_type=full_name_with_name(klass=type(self)),
            use_tensors=use_tensors,
            arrays=arrays,
            tensors=tensors,
        )

    @staticmethod
    def _proto2object(proto: Tensor_PB) -> RowEntityPhiTensor:
        use_tensors = proto.use_tensors
        child = []
        if use_tensors:
            child = [deserialize(tensor) for tensor in proto.tensors]
        else:
            child = [deserialize(array) for array in proto.arrays]

        return RowEntityPhiTensor(child)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB


@implements(RowEntityPhiTensor, np.expand_dims)
def expand_dims(a, axis):

    if axis == 0:
        raise Exception(
            "Currently, we don't have functionality for axis=0 but we could with a bit more work."
        )

    new_rows = list()
    for row in a.child:
        new_rows.append(np.expand_dims(row, axis - 1))

    return RowEntityPhiTensor(rows=new_rows, check_shape=False)
