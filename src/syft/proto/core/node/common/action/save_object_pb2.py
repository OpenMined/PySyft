# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/action/save_object.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2
from syft.proto.core.store import store_object_pb2 as proto_dot_core_dot_store_dot_store__object__pb2
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/core/node/common/action/save_object.proto',
  package='syft.core.node.common.action',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n/proto/core/node/common/action/save_object.proto\x12\x1csyft.core.node.common.action\x1a%proto/core/common/common_object.proto\x1a#proto/core/store/store_object.proto\x1a\x1bproto/core/io/address.proto\"\xbb\x01\n\x10SaveObjectAction\x12-\n\x0eid_at_location\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12,\n\x03obj\x18\x02 \x01(\x0b\x32\x1f.syft.core.store.StorableObject\x12&\n\x07\x61\x64\x64ress\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address\x12\"\n\x1a\x61nyone_can_search_for_this\x18\x04 \x01(\x08\x62\x06proto3'
  ,
  dependencies=[proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,proto_dot_core_dot_store_dot_store__object__pb2.DESCRIPTOR,proto_dot_core_dot_io_dot_address__pb2.DESCRIPTOR,])




_SAVEOBJECTACTION = _descriptor.Descriptor(
  name='SaveObjectAction',
  full_name='syft.core.node.common.action.SaveObjectAction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id_at_location', full_name='syft.core.node.common.action.SaveObjectAction.id_at_location', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='obj', full_name='syft.core.node.common.action.SaveObjectAction.obj', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.core.node.common.action.SaveObjectAction.address', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anyone_can_search_for_this', full_name='syft.core.node.common.action.SaveObjectAction.anyone_can_search_for_this', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=187,
  serialized_end=374,
)

_SAVEOBJECTACTION.fields_by_name['id_at_location'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_SAVEOBJECTACTION.fields_by_name['obj'].message_type = proto_dot_core_dot_store_dot_store__object__pb2._STORABLEOBJECT
_SAVEOBJECTACTION.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
DESCRIPTOR.message_types_by_name['SaveObjectAction'] = _SAVEOBJECTACTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SaveObjectAction = _reflection.GeneratedProtocolMessageType('SaveObjectAction', (_message.Message,), {
  'DESCRIPTOR' : _SAVEOBJECTACTION,
  '__module__' : 'proto.core.node.common.action.save_object_pb2'
  # @@protoc_insertion_point(class_scope:syft.core.node.common.action.SaveObjectAction)
  })
_sym_db.RegisterMessage(SaveObjectAction)


# @@protoc_insertion_point(module_scope)
