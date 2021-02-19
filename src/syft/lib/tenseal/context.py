# stdlib
import sys
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from packaging import version
import tenseal as ts

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...logger import info
from ...logger import traceback_and_raise
from ...proto.util.vendor_bytes_pb2 import VendorBytes as VendorBytes_PB
from ...util import aggressive_set_attr
from ...util import get_fully_qualified_name


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
        proto.obj_type = get_fully_qualified_name(obj=self.value)
        proto.vendor_lib = "tenseal"
        proto.vendor_lib_version = ts.__version__
        # TODO: sending the secret key reduces the latency by a lot, but
        # we need strong guarantees that it won't be misused.
        proto.content = self.value.serialize(save_secret_key=True)  # type: ignore

        return proto

    @staticmethod
    def _data_proto2object(proto: VendorBytes_PB) -> ts.Context:
        vendor_lib = proto.vendor_lib
        lib_version = version.parse(proto.vendor_lib_version)

        if vendor_lib not in sys.modules:
            traceback_and_raise(
                Exception(
                    f"{vendor_lib} version: {proto.vendor_lib_version} is required"
                )
            )
        else:
            if lib_version > version.parse(ts.__version__):
                log = f"Warning {lib_version} > local imported version {ts.__version__}"
                info(log)

        # TODO: Here we need to generate all the necessary public keys and drop the secret key.
        # Right now, the context is serialized again, and we lose all the performance improvements.
        return ts.context_from(proto.content, n_threads=1)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VendorBytes_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return ts.Context

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


class SchemeTypeWrapper(StorableObject):
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
        proto.obj_type = get_fully_qualified_name(obj=self.value)
        proto.vendor_lib = "tenseal"
        proto.vendor_lib_version = ts.__version__
        proto.content = str.encode(self.value.name)  # type: ignore

        return proto

    @staticmethod
    def _data_proto2object(proto: VendorBytes_PB) -> ts.SCHEME_TYPE:
        vendor_lib = proto.vendor_lib
        lib_version = version.parse(proto.vendor_lib_version)

        if vendor_lib not in sys.modules:
            traceback_and_raise(
                Exception(
                    f"{vendor_lib} version: {proto.vendor_lib_version} is required"
                )
            )
        else:
            if lib_version > version.parse(ts.__version__):
                log = f"Warning {lib_version} > local imported version {ts.__version__}"
                info(log)

        return ts.SCHEME_TYPE[proto.content.decode()]

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VendorBytes_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return ts.SCHEME_TYPE

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
    obj=ts.Context, name="serializable_wrapper_type", attr=ContextWrapper
)

aggressive_set_attr(
    obj=ts.SCHEME_TYPE, name="serializable_wrapper_type", attr=SchemeTypeWrapper
)
