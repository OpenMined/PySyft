# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/adp/entity.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1bproto/core/adp/entity.proto\x12\rsyft.core.adp\x1a%proto/core/common/common_object.proto"9\n\x06\x45ntity\x12\x0c\n\x04name\x18\x01 \x01(\t\x12!\n\x02id\x18\x02 \x01(\x0b\x32\x15.syft.core.common.UID"^\n\x10\x44\x61taSubjectGroup\x12!\n\x02id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\'\n\x08\x65ntities\x18\x02 \x03(\x0b\x32\x15.syft.core.adp.Entityb\x06proto3'
)


_ENTITY = DESCRIPTOR.message_types_by_name["Entity"]
_DATASUBJECTGROUP = DESCRIPTOR.message_types_by_name["DataSubjectGroup"]
Entity = _reflection.GeneratedProtocolMessageType(
    "Entity",
    (_message.Message,),
    {
        "DESCRIPTOR": _ENTITY,
        "__module__": "proto.core.adp.entity_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.adp.Entity)
    },
)
_sym_db.RegisterMessage(Entity)

DataSubjectGroup = _reflection.GeneratedProtocolMessageType(
    "DataSubjectGroup",
    (_message.Message,),
    {
        "DESCRIPTOR": _DATASUBJECTGROUP,
        "__module__": "proto.core.adp.entity_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.adp.DataSubjectGroup)
    },
)
_sym_db.RegisterMessage(DataSubjectGroup)

if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _ENTITY._serialized_start = 85
    _ENTITY._serialized_end = 142
    _DATASUBJECTGROUP._serialized_start = 144
    _DATASUBJECTGROUP._serialized_end = 238
# @@protoc_insertion_point(module_scope)
