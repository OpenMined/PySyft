# third party
import tenseal as ts

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.tenseal.vector_pb2 import TenSEALVector as TenSEALVector_PB
from ..util import full_name_with_name


def object2proto(obj: object) -> TenSEALVector_PB:
    proto = TenSEALVector_PB()
    proto.obj_type = full_name_with_name(klass=obj._sy_serializable_wrapper_type)  # type: ignore
    proto.vector = obj.serialize()  # type: ignore

    return proto


def proto2object(proto: TenSEALVector_PB) -> ts.CKKSVector:
    vec = ts.lazy_ckks_vector_from(proto.vector)

    return vec


GenerateWrapper(
    wrapped_type=ts.CKKSVector,
    import_path="tenseal.CKKSVector",
    protobuf_scheme=TenSEALVector_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
