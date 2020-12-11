# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import tenseal as ts

# syft relative
from ...core.common.serde.deserialize import _deserialize
from ...core.common.serde.serialize import _serialize
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.tenseal.vector_pb2 import TenSEALVector as TenSEALVector_PB
from ...util import aggressive_set_attr
from ...util import get_fully_qualified_name


class CKKSVector(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> TenSEALVector_PB:
        proto = TenSEALVector_PB()
        proto.id.CopyFrom(_serialize(obj=self.id))
        proto.obj_type = get_fully_qualified_name(obj=self.value)
        proto.context = self.value.context().serialize()  # type: ignore
        proto.vector = self.value.serialize()  # type: ignore

        return proto

    @staticmethod
    def _data_proto2object(proto: TenSEALVector_PB) -> ts.CKKSVector:
        vec_id: UID = _deserialize(blob=proto.id)
        context = ts.context_from(proto.context)
        vec = ts.ckks_vector_from(context, proto.vector)
        vec.id = vec_id

        return vec

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return TenSEALVector_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return ts.CKKSVector

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=ts.CKKSVector, name="serializable_wrapper_type", attr=CKKSVector
)
