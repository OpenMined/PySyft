# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/lib/python/dict.proto
"""Generated protocol buffer code."""
# third party
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# syft absolute
from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1bproto/lib/python/dict.proto\x12\x0fsyft.lib.python\x1a%proto/core/common/common_object.proto"^\n\x04\x44ict\x12\x0c\n\x04keys\x18\x01 \x03(\x0c\x12\x0e\n\x06values\x18\x02 \x03(\x0c\x12!\n\x02id\x18\x03 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x15\n\rtemporary_box\x18\x04 \x01(\x08\x62\x06proto3'
)


_DICT = DESCRIPTOR.message_types_by_name["Dict"]
Dict = _reflection.GeneratedProtocolMessageType(
    "Dict",
    (_message.Message,),
    {
        "DESCRIPTOR": _DICT,
        "__module__": "proto.lib.python.dict_pb2"
        # @@protoc_insertion_point(class_scope:syft.lib.python.Dict)
    },
)
_sym_db.RegisterMessage(Dict)

if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _DICT._serialized_start = 87
    _DICT._serialized_end = 181
# @@protoc_insertion_point(module_scope)
