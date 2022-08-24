# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.nacl.verify_key_pb2 import VerifyKeyProto


def serialize_verify_key(obj: VerifyKey) -> VerifyKeyProto:
    return VerifyKeyProto(data=obj.encode(encoder=HexEncoder))


def deserialize_verify_key(proto: VerifyKeyProto) -> VerifyKey:
    return VerifyKey(proto.data, encoder=HexEncoder)


serializable(generate_wrapper=True)(
    wrapped_type=VerifyKey,
    import_path="nacl.signing.VerifyKey",
    protobuf_scheme=VerifyKeyProto,
    type_object2proto=serialize_verify_key,
    type_proto2object=deserialize_verify_key,
)
