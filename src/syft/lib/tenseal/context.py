# stdlib
import sys

# third party
from packaging import version
import tenseal as ts

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...logger import info
from ...logger import traceback_and_raise
from ...proto.util.vendor_bytes_pb2 import VendorBytes as VendorBytes_PB
from ...util import get_fully_qualified_name
from ..util import full_name_with_name


def context_object2proto(obj: object) -> VendorBytes_PB:
    proto = VendorBytes_PB()
    proto.obj_type = full_name_with_name(klass=obj._sy_serializable_wrapper_type)  # type: ignore
    proto.vendor_lib = "tenseal"
    proto.vendor_lib_version = ts.__version__
    proto.content = obj.serialize(save_secret_key=True)  # type: ignore

    return proto


def context_proto2object(proto: VendorBytes_PB) -> ts.Context:
    vendor_lib = proto.vendor_lib
    lib_version = version.parse(proto.vendor_lib_version)

    if vendor_lib not in sys.modules:
        traceback_and_raise(
            Exception(f"{vendor_lib} version: {proto.vendor_lib_version} is required")
        )
    else:
        if lib_version > version.parse(ts.__version__):
            log = f"Warning {lib_version} > local imported version {ts.__version__}"
            info(log)

    return ts.context_from(proto.content, n_threads=1)


GenerateWrapper(
    wrapped_type=ts.Context,
    import_path="tenseal.Context",
    protobuf_scheme=VendorBytes_PB,
    type_object2proto=context_object2proto,
    type_proto2object=context_proto2object,
)


def schemetype_object2proto(obj: object) -> VendorBytes_PB:
    proto = VendorBytes_PB()
    proto.obj_type = get_fully_qualified_name(obj=obj)
    proto.vendor_lib = "tenseal"
    proto.vendor_lib_version = ts.__version__
    proto.content = str.encode(obj.name)  # type: ignore

    return proto


def schemetype_proto2object(proto: VendorBytes_PB) -> ts.SCHEME_TYPE:
    vendor_lib = proto.vendor_lib
    lib_version = version.parse(proto.vendor_lib_version)

    if vendor_lib not in sys.modules:
        traceback_and_raise(
            Exception(f"{vendor_lib} version: {proto.vendor_lib_version} is required")
        )
    else:
        if lib_version > version.parse(ts.__version__):
            log = f"Warning {lib_version} > local imported version {ts.__version__}"
            info(log)

    return ts.SCHEME_TYPE[proto.content.decode()]


GenerateWrapper(
    wrapped_type=ts.SCHEME_TYPE,
    import_path="tenseal.SCHEME_TYPE",
    protobuf_scheme=VendorBytes_PB,
    type_object2proto=schemetype_object2proto,
    type_proto2object=schemetype_proto2object,
)
