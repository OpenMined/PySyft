# stdlib
import sys
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from packaging import version
import tenseal as ts

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...proto.util.vendor_bytes_pb2 import VendorBytes as VendorBytes_PB
from ...util import aggressive_set_attr
from ...util import get_fully_qualified_name

context_type = ts._ts_cpp.TenSEALContext


class ContextWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> VendorBytes_PB:
        proto = VendorBytes_PB()
        proto.obj_type = get_fully_qualified_name(obj=self)
        proto.vendor_lib = "tenseal"
        proto.vendor_lib_version = ts.__version__
        proto.bytes = self.serialize()

        return proto

    @staticmethod
    def _data_proto2object(proto: VendorBytes_PB) -> context_type:  # type: ignore
        vendor_lib = proto.vendor_lib
        lib_version = version.parse(proto.vendor_lib_version)

        if vendor_lib not in sys.modules:
            raise Exception(
                f"{vendor_lib} version: {proto.vendor_lib_version} is required"
            )
        else:
            if lib_version > version.parse(ts.__version__):
                log = f"Warning {lib_version} > local imported version {ts.__version__}"
                print(log)
                logger.info(log)

        return ts.context_from(proto.bytes)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VendorBytes_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return context_type

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
    obj=context_type, name="serializable_wrapper_type", attr=ContextWrapper
)
