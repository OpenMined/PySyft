# stdlib
from collections import namedtuple
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch

# syft relative
from ...core.common.serde.deserialize import _deserialize
from ...core.common.serde.serializable import Serializable
from ...core.common.serde.serialize import _serialize
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...lib.util import full_name_with_qualname
from ...proto.lib.torch.valuesindices_pb2 import ValuesIndicesProto as ValuesIndices_PB
from ...util import aggressive_set_attr
from ..torch.tensor_util import protobuf_tensor_deserializer
from ..torch.tensor_util import protobuf_tensor_serializer

# this is all the different named tuple attrs so they can be used if an object doesnt
# have them then getting the wrong attr will fail this needs to be improved with unions
# or specific types for every single torch.return_types.* namedtuple
all_attrs = tuple(
    [
        "values",
        "indices",
        "eigenvalues",
        "eigenvectors",
        "solution",
        "QR",
        "sign",
        "logabsdet",
        "Q",
        "R",
        "LU",
        "cloned_coefficient",
        "U",
        "S",
        "V",
        "a",
        "tau",
    ]
)

# this is a dummy type for our ValuesIndicesPointer
ValuesIndices = namedtuple("ValuesIndices", all_attrs)  # type: ignore


class ValuesIndicesWrapper(StorableObject):
    def __init__(self, value: object):
        _id = getattr(value, "id", UID())
        obj_type, values = ValuesIndicesWrapper.get_parts(return_tuple=value)
        return_tuple = ValuesIndicesWrapper.make_namedtuple(
            obj_type=obj_type, values=values, id=_id
        )

        super().__init__(
            data=return_tuple,
            id=_id,
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )

        self.value = return_tuple

    def _data_object2proto(self) -> ValuesIndices_PB:
        obj_type = full_name_with_qualname(klass=type(self.data))
        keys = ValuesIndicesWrapper.get_keys(klass_name=obj_type)

        values = []
        for key in keys:
            values.append(getattr(self.data, key, None))

        proto = ValuesIndices_PB()
        proto.values.extend(list(map(lambda x: protobuf_tensor_serializer(x), values)))
        proto.keys.extend(list(keys))
        # proto.values.CopyFrom(protobuf_tensor_serializer(values))
        # proto.indices.CopyFrom(protobuf_tensor_serializer(indices))
        proto.id.CopyFrom(_serialize(obj=self.id))
        proto.obj_type = obj_type

        return proto

    @staticmethod
    def _data_proto2object(proto: ValuesIndices_PB) -> "ValuesIndices":  # type: ignore
        _id: UID = _deserialize(blob=proto.id)
        values = [protobuf_tensor_deserializer(x) for x in proto.values]
        # keys = proto.keys

        return_type = ValuesIndicesWrapper.make_namedtuple(
            obj_type=proto.obj_type, values=values, id=_id
        )

        return return_type

    @staticmethod
    def get_keys(klass_name: str) -> List[str]:
        keys = []

        if klass_name == "torch.return_types.eig":
            key1 = "eigenvalues"
            key2 = "eigenvectors"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.lstsq":
            key1 = "solution"
            key2 = "QR"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.slogdet":
            key1 = "sign"
            key2 = "logabsdet"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.qr":
            key1 = "Q"
            key2 = "R"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.solve":
            key1 = "solution"
            key2 = "LU"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.symeig":
            key1 = "eigenvalues"
            key2 = "eigenvectors"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.triangular_solve":
            key1 = "solution"
            key2 = "cloned_coefficient"
            keys.append(key1)
            keys.append(key2)
        elif klass_name == "torch.return_types.svd":
            key1 = "U"
            key2 = "S"
            key3 = "V"
            keys.append(key1)
            keys.append(key2)
            keys.append(key3)
        elif klass_name == "torch.return_types.geqrf":
            key1 = "a"
            key2 = "tau"
            keys.append(key1)
            keys.append(key2)
        else:
            # default
            key1 = "values"
            key2 = "indices"
            keys.append(key1)
            keys.append(key2)

        return keys

    @staticmethod
    def get_parts(return_tuple: Any) -> Tuple[str, List[torch.Tensor]]:
        obj_type = full_name_with_qualname(klass=type(return_tuple))
        keys = ValuesIndicesWrapper.get_keys(klass_name=obj_type)
        values = []
        for key in keys:
            values.append(getattr(return_tuple, key))

        return (obj_type, values)

    @staticmethod
    def make_namedtuple(
        obj_type: str,
        values: List[torch.Tensor],
        id: UID,
        tags: List[str] = [],
        description: str = "",
    ) -> Any:
        module_parts = obj_type.split(".")
        klass = module_parts.pop()
        module_name = ".".join(module_parts)
        keys = ValuesIndicesWrapper.get_keys(klass_name=obj_type)
        tuple_klass = namedtuple(  # type: ignore
            klass, (*keys, "tags", "description", "id")
        )
        tuple_klass.__module__ = module_name
        return tuple_klass(*values, tags, description, id)  # type: ignore

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return ValuesIndices_PB

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        obj_type, values = ValuesIndicesWrapper.get_parts(return_tuple=data)

        if tags is None:
            # for the type checker
            tags = []
        if description is None:
            # for the type checker
            description = ""

        return_tuple = ValuesIndicesWrapper.make_namedtuple(
            obj_type=obj_type,
            values=values,
            id=id,
            tags=tags,
            description=description,
        )

        return return_tuple


# get each of the dynamic torch.return_types.*
def add_torch_return_types() -> None:
    supported_types = []
    A = torch.tensor([[1.0, 1, 1], [2, 3, 4], [3, 5, 2], [4, 2, 5], [5, 4, 3]])
    B = torch.tensor([[-10.0, -3], [12, 14], [14, 12], [16, 16], [18, 16]])
    x = torch.Tensor([[1, 2], [1, 2]])
    s = torch.tensor(
        [[-0.1000, 0.1000, 0.2000], [0.2000, 0.3000, 0.4000], [0.0000, -0.3000, 0.5000]]
    )

    cummax = x.cummax(0)
    supported_types.append(type(cummax))

    cummin = x.cummin(0)
    supported_types.append(type(cummin))

    eig = x.eig(True)
    supported_types.append(type(eig))

    kthvalue = x.kthvalue(1)
    supported_types.append(type(kthvalue))

    lstsq = A.lstsq(B)
    supported_types.append(type(lstsq))

    slogdet = x.slogdet()
    supported_types.append(type(slogdet))

    qr = x.qr()
    supported_types.append(type(qr))

    mode = x.mode()
    supported_types.append(type(mode))

    solve = s.solve(s)
    supported_types.append(type(solve))

    sort = s.sort()
    supported_types.append(type(sort))

    symeig = s.symeig()
    supported_types.append(type(symeig))

    topk = s.topk(1)
    supported_types.append(type(topk))

    triangular_solve = s.triangular_solve(s)
    supported_types.append(type(triangular_solve))

    svd = s.svd()
    supported_types.append(type(svd))

    geqrf = s.geqrf()
    supported_types.append(type(geqrf))

    median = s.median(0)
    supported_types.append(type(median))

    max_t = s.max(0)
    supported_types.append(type(max_t))

    min_t = s.min(0)
    supported_types.append(type(min_t))

    for types in supported_types:
        aggressive_set_attr(
            obj=types, name="serializable_wrapper_type", attr=ValuesIndicesWrapper
        )

        def attr_serialize(  # type: ignore
            self,
            to_proto: bool = True,
            to_bytes: bool = False,
        ) -> Union[str, bytes, Message]:
            return _serialize(
                obj=self,
                to_proto=to_proto,
                to_bytes=to_bytes,
            )

        aggressive_set_attr(obj=types, name="serialize", attr=attr_serialize)
        aggressive_set_attr(obj=types, name="to_proto", attr=Serializable.to_proto)
        aggressive_set_attr(obj=types, name="proto", attr=Serializable.proto)
        aggressive_set_attr(obj=types, name="to_bytes", attr=Serializable.to_bytes)


add_torch_return_types()
