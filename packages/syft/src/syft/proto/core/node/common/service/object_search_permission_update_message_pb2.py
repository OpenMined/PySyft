# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/service/object_search_permission_update_message.proto
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
    b'\nLproto/core/node/common/service/object_search_permission_update_message.proto\x12\x1dsyft.core.node.common.service\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto"\xdf\x01\n#ObjectSearchPermissionUpdateMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x19\n\x11target_verify_key\x18\x03 \x01(\x0c\x12/\n\x10target_object_id\x18\x04 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x1d\n\x15\x61\x64\x64_instead_of_remove\x18\x05 \x01(\x08\x62\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR,
    "proto.core.node.common.service.object_search_permission_update_message_pb2",
    globals(),
)
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _OBJECTSEARCHPERMISSIONUPDATEMESSAGE._serialized_start = 180
    _OBJECTSEARCHPERMISSIONUPDATEMESSAGE._serialized_end = 403
# @@protoc_insertion_point(module_scope)
