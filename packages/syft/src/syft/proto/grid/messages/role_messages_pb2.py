# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/grid/messages/role_messages.proto
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
from syft.proto.lib.python import dict_pb2 as proto_dot_lib_dot_python_dot_dict__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\'proto/grid/messages/role_messages.proto\x12\x12syft.grid.messages\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto\x1a\x1bproto/lib/python/dict.proto"\xcb\x03\n\x11\x43reateRoleMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x1e\n\x16\x63\x61n_make_data_requests\x18\x04 \x01(\x08\x12 \n\x18\x63\x61n_triage_data_requests\x18\x05 \x01(\x08\x12!\n\x19\x63\x61n_manage_privacy_budget\x18\x06 \x01(\x08\x12\x18\n\x10\x63\x61n_create_users\x18\x07 \x01(\x08\x12\x18\n\x10\x63\x61n_manage_users\x18\x08 \x01(\x08\x12\x16\n\x0e\x63\x61n_edit_roles\x18\t \x01(\x08\x12!\n\x19\x63\x61n_manage_infrastructure\x18\n \x01(\x08\x12\x17\n\x0f\x63\x61n_upload_data\x18\x0b \x01(\x08\x12!\n\x19\x63\x61n_upload_legal_document\x18\x0c \x01(\x08\x12 \n\x18\x63\x61n_edit_domain_settings\x18\r \x01(\x08\x12\'\n\x08reply_to\x18\x0e \x01(\x0b\x32\x15.syft.core.io.Address"\x99\x01\n\x0eGetRoleMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07role_id\x18\x03 \x01(\x05\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x88\x01\n\x0fGetRoleResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x63ontent\x18\x02 \x01(\x0b\x32\x15.syft.lib.python.Dict\x12&\n\x07\x61\x64\x64ress\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address"\x89\x01\n\x0fGetRolesMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x89\x01\n\x10GetRolesResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x63ontent\x18\x02 \x03(\x0b\x32\x15.syft.lib.python.Dict\x12&\n\x07\x61\x64\x64ress\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address"\xdc\x03\n\x11UpdateRoleMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x1e\n\x16\x63\x61n_make_data_requests\x18\x04 \x01(\x08\x12 \n\x18\x63\x61n_triage_data_requests\x18\x05 \x01(\x08\x12!\n\x19\x63\x61n_manage_privacy_budget\x18\x06 \x01(\x08\x12\x18\n\x10\x63\x61n_create_users\x18\x07 \x01(\x08\x12\x18\n\x10\x63\x61n_manage_users\x18\x08 \x01(\x08\x12\x16\n\x0e\x63\x61n_edit_roles\x18\t \x01(\x08\x12!\n\x19\x63\x61n_manage_infrastructure\x18\n \x01(\x08\x12\x17\n\x0f\x63\x61n_upload_data\x18\x0b \x01(\x08\x12!\n\x19\x63\x61n_upload_legal_document\x18\x0c \x01(\x08\x12 \n\x18\x63\x61n_edit_domain_settings\x18\r \x01(\x08\x12\x0f\n\x07role_id\x18\x0e \x01(\x05\x12\'\n\x08reply_to\x18\x0f \x01(\x0b\x32\x15.syft.core.io.Address"\x9c\x01\n\x11\x44\x65leteRoleMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07role_id\x18\x03 \x01(\x05\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Addressb\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "proto.grid.messages.role_messages_pb2", globals()
)
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _CREATEROLEMESSAGE._serialized_start = 161
    _CREATEROLEMESSAGE._serialized_end = 620
    _GETROLEMESSAGE._serialized_start = 623
    _GETROLEMESSAGE._serialized_end = 776
    _GETROLERESPONSE._serialized_start = 779
    _GETROLERESPONSE._serialized_end = 915
    _GETROLESMESSAGE._serialized_start = 918
    _GETROLESMESSAGE._serialized_end = 1055
    _GETROLESRESPONSE._serialized_start = 1058
    _GETROLESRESPONSE._serialized_end = 1195
    _UPDATEROLEMESSAGE._serialized_start = 1198
    _UPDATEROLEMESSAGE._serialized_end = 1674
    _DELETEROLEMESSAGE._serialized_start = 1677
    _DELETEROLEMESSAGE._serialized_end = 1833
# @@protoc_insertion_point(module_scope)
