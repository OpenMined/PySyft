# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.nacl.signing_key_pb2 import SigningKeyProto


def serialize_sign_key(obj: SigningKey) -> SigningKeyProto:
    return SigningKeyProto(data=obj.encode(encoder=HexEncoder))


def deserialize_sign_key(proto: SigningKeyProto) -> SigningKey:
    return SigningKey(proto.data, encoder=HexEncoder)


serializable(generate_wrapper=True)(
    wrapped_type=SigningKey,
    import_path="nacl.signing.SigningKey",
    protobuf_scheme=SigningKeyProto,
    type_object2proto=serialize_sign_key,
    type_proto2object=deserialize_sign_key,
)
