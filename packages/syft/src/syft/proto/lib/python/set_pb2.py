# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/lib/python/set.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1aproto/lib/python/set.proto\x12\x0fsyft.lib.python\x1a%proto/core/common/common_object.proto"6\n\x03Set\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x0c\x12!\n\x02id\x18\x02 \x01(\x0b\x32\x15.syft.core.common.UIDb\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "proto.lib.python.set_pb2", globals()
)
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SET._serialized_start = 86
    _SET._serialized_end = 140
# @@protoc_insertion_point(module_scope)
