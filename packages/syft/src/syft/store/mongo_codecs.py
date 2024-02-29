# stdlib
from typing import Any

# third party
from bson import CodecOptions
from bson.binary import Binary
from bson.binary import USER_DEFINED_SUBTYPE
from bson.codec_options import TypeDecoder
from bson.codec_options import TypeRegistry

# relative
from ..serde.deserialize import _deserialize
from ..serde.serialize import _serialize


def fallback_syft_encoder(value: object) -> Binary:
    return Binary(_serialize(value, to_bytes=True), USER_DEFINED_SUBTYPE)


class SyftMongoBinaryDecoder(TypeDecoder):
    bson_type = Binary

    def transform_bson(self, value: Any) -> Any:
        if value.subtype == USER_DEFINED_SUBTYPE:
            return _deserialize(value, from_bytes=True)
        return value


syft_codecs = [SyftMongoBinaryDecoder()]
syft_type_registry = TypeRegistry(syft_codecs, fallback_encoder=fallback_syft_encoder)
SYFT_CODEC_OPTIONS = CodecOptions(type_registry=syft_type_registry)
