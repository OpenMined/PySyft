# stdlib
from typing import Any
from typing import ClassVar
from typing import Type

# third party
import numpy as np

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from .action_object import ActionObject
from .action_object import BASE_PASSTHROUGH_ATTRS
from .action_types import action_types

# @serializable(attrs=["id", "node_uid", "parent_id"])
# class NumpyArrayObjectPointer(ActionObjectPointer):
#     _inflix_operations = ["__add__", "__sub__", "__eq__", "__mul__"]
#     __canonical_name__ = "NumpyArrayObjectPointer"
#     __version__ = SYFT_OBJECT_VERSION_1

#     def get_from(self, domain_client) -> Any:
#         return domain_client.api.services.action.get(self.id).syft_action_data


class NumpyArrayObjectPointer:
    pass


def numpy_like_eq(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, bool):
        return result

    if hasattr(result, "all"):
        return (result).all()
    return bool(result)


# 🔵 TODO 7: Map TPActionObjects and their 3rd Party types like numpy type to these
# classes for bi-directional lookup.
@serializable()
class NumpyArrayObject(ActionObject, np.lib.mixins.NDArrayOperatorsMixin):
    __canonical_name__ = "NumpyArrayObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[Type[Any]] = np.ndarray
    syft_pointer_type = NumpyArrayObjectPointer
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
    syft_dont_wrap_attrs = ["dtype", "shape"]

    # def __eq__(self, other: Any) -> bool:
    #     # 🟡 TODO 8: move __eq__ to a Data / Serdeable type interface on ActionObject
    #     if isinstance(other, NumpyArrayObject):
    #         return (
    #             numpy_like_eq(self.syft_action_data, other.syft_action_data)
    #             and self.syft_pointer_type == other.syft_pointer_type
    #         )
    #     return self == other

    def __bool__(self) -> bool:
        return bool(self.all())

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(
            np.array(x.syft_action_data, dtype=x.dtype)
            if isinstance(x, NumpyArrayObject)
            else x
            for x in inputs
        )

        result = getattr(ufunc, method)(*inputs, **kwargs)
        if type(result) is tuple:
            return tuple(
                NumpyArrayObject(syft_action_data=x, dtype=x.dtype, shape=x.shape)
                for x in result
            )
        else:
            return NumpyArrayObject(
                syft_action_data=result, dtype=result.dtype, shape=result.shape
            )


@serializable()
class NumpyScalarObject(ActionObject, np.lib.mixins.NDArrayOperatorsMixin):
    __canonical_name__ = "NumpyScalarObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = np.number
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
    syft_dont_wrap_attrs = ["dtype", "shape"]

    def __float__(self) -> float:
        return float(self.syft_action_data)


@serializable()
class NumpyBoolObject(ActionObject, np.lib.mixins.NDArrayOperatorsMixin):
    __canonical_name__ = "NumpyBoolObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = np.bool_
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
    syft_dont_wrap_attrs = ["dtype", "shape"]


np_array = np.array([1, 2, 3])
action_types[type(np_array)] = NumpyArrayObject


SUPPORTED_BOOL_TYPES = [np.bool_]

for scalar_type in SUPPORTED_BOOL_TYPES:
    action_types[scalar_type] = NumpyBoolObject


SUPPORTED_INT_TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

SUPPORTED_FLOAT_TYPES = [
    np.float16,
    np.float32,
    np.float64,
]

for scalar_type in SUPPORTED_INT_TYPES + SUPPORTED_FLOAT_TYPES:
    action_types[scalar_type] = NumpyScalarObject
