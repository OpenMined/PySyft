# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/action/garbage_collect_object.proto
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
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n:proto/core/node/common/action/garbage_collect_object.proto\x12\x1csyft.core.node.common.action\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto"s\n\x1aGarbageCollectObjectAction\x12-\n\x0eid_at_location\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Addressb\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "proto.core.node.common.action.garbage_collect_object_pb2", globals()
)
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _GARBAGECOLLECTOBJECTACTION._serialized_start = 160
    _GARBAGECOLLECTOBJECTACTION._serialized_end = 275
# @@protoc_insertion_point(module_scope)
