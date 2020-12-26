# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/grid/service/signaling_service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2
from syft.proto.core.node.common import metadata_pb2 as proto_dot_core_dot_node_dot_common_dot_metadata__pb2
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/grid/service/signaling_service.proto',
  package='syft.grid.service',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n*proto/grid/service/signaling_service.proto\x12\x11syft.grid.service\x1a%proto/core/common/common_object.proto\x1a%proto/core/node/common/metadata.proto\x1a\x1bproto/core/io/address.proto\"\x90\x01\n\x16RegisterNewPeerMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\'\n\x08reply_to\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address\"|\n\x1aPeerSuccessfullyRegistered\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07peer_id\x18\x03 \x01(\t\"\xd8\x01\n\x16SignalingAnswerMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07payload\x18\x03 \x01(\t\x12\x36\n\rhost_metadata\x18\x04 \x01(\x0b\x32\x1f.syft.core.node.common.Metadata\x12\x13\n\x0btarget_peer\x18\x05 \x01(\t\x12\x11\n\thost_peer\x18\x06 \x01(\t\"\xd7\x01\n\x15SignalingOfferMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07payload\x18\x03 \x01(\t\x12\x36\n\rhost_metadata\x18\x04 \x01(\x0b\x32\x1f.syft.core.node.common.Metadata\x12\x13\n\x0btarget_peer\x18\x05 \x01(\t\x12\x11\n\thost_peer\x18\x06 \x01(\t\"\xb9\x01\n\x17OfferPullRequestMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\'\n\x08reply_to\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x13\n\x0btarget_peer\x18\x04 \x01(\t\x12\x11\n\thost_peer\x18\x05 \x01(\t\"\xba\x01\n\x18\x41nswerPullRequestMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\'\n\x08reply_to\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x13\n\x0btarget_peer\x18\x04 \x01(\t\x12\x11\n\thost_peer\x18\x05 \x01(\t\"j\n\x19SignalingRequestsNotFound\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\"g\n\x16InvalidLoopBackRequest\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\"g\n\x16\x43loseConnectionMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Addressb\x06proto3'
  ,
  dependencies=[proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,proto_dot_core_dot_node_dot_common_dot_metadata__pb2.DESCRIPTOR,proto_dot_core_dot_io_dot_address__pb2.DESCRIPTOR,])




_REGISTERNEWPEERMESSAGE = _descriptor.Descriptor(
  name='RegisterNewPeerMessage',
  full_name='syft.grid.service.RegisterNewPeerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.RegisterNewPeerMessage.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.RegisterNewPeerMessage.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reply_to', full_name='syft.grid.service.RegisterNewPeerMessage.reply_to', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=173,
  serialized_end=317,
)


_PEERSUCCESSFULLYREGISTERED = _descriptor.Descriptor(
  name='PeerSuccessfullyRegistered',
  full_name='syft.grid.service.PeerSuccessfullyRegistered',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.PeerSuccessfullyRegistered.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.PeerSuccessfullyRegistered.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='peer_id', full_name='syft.grid.service.PeerSuccessfullyRegistered.peer_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=319,
  serialized_end=443,
)


_SIGNALINGANSWERMESSAGE = _descriptor.Descriptor(
  name='SignalingAnswerMessage',
  full_name='syft.grid.service.SignalingAnswerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.SignalingAnswerMessage.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.SignalingAnswerMessage.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='syft.grid.service.SignalingAnswerMessage.payload', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_metadata', full_name='syft.grid.service.SignalingAnswerMessage.host_metadata', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='target_peer', full_name='syft.grid.service.SignalingAnswerMessage.target_peer', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_peer', full_name='syft.grid.service.SignalingAnswerMessage.host_peer', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=446,
  serialized_end=662,
)


_SIGNALINGOFFERMESSAGE = _descriptor.Descriptor(
  name='SignalingOfferMessage',
  full_name='syft.grid.service.SignalingOfferMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.SignalingOfferMessage.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.SignalingOfferMessage.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='syft.grid.service.SignalingOfferMessage.payload', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_metadata', full_name='syft.grid.service.SignalingOfferMessage.host_metadata', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='target_peer', full_name='syft.grid.service.SignalingOfferMessage.target_peer', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_peer', full_name='syft.grid.service.SignalingOfferMessage.host_peer', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=665,
  serialized_end=880,
)


_OFFERPULLREQUESTMESSAGE = _descriptor.Descriptor(
  name='OfferPullRequestMessage',
  full_name='syft.grid.service.OfferPullRequestMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.OfferPullRequestMessage.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.OfferPullRequestMessage.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reply_to', full_name='syft.grid.service.OfferPullRequestMessage.reply_to', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='target_peer', full_name='syft.grid.service.OfferPullRequestMessage.target_peer', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_peer', full_name='syft.grid.service.OfferPullRequestMessage.host_peer', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=883,
  serialized_end=1068,
)


_ANSWERPULLREQUESTMESSAGE = _descriptor.Descriptor(
  name='AnswerPullRequestMessage',
  full_name='syft.grid.service.AnswerPullRequestMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.AnswerPullRequestMessage.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.AnswerPullRequestMessage.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reply_to', full_name='syft.grid.service.AnswerPullRequestMessage.reply_to', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='target_peer', full_name='syft.grid.service.AnswerPullRequestMessage.target_peer', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_peer', full_name='syft.grid.service.AnswerPullRequestMessage.host_peer', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=1071,
  serialized_end=1257,
)


_SIGNALINGREQUESTSNOTFOUND = _descriptor.Descriptor(
  name='SignalingRequestsNotFound',
  full_name='syft.grid.service.SignalingRequestsNotFound',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.SignalingRequestsNotFound.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.SignalingRequestsNotFound.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=1259,
  serialized_end=1365,
)


_INVALIDLOOPBACKREQUEST = _descriptor.Descriptor(
  name='InvalidLoopBackRequest',
  full_name='syft.grid.service.InvalidLoopBackRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.InvalidLoopBackRequest.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.InvalidLoopBackRequest.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=1367,
  serialized_end=1470,
)


_CLOSECONNECTIONMESSAGE = _descriptor.Descriptor(
  name='CloseConnectionMessage',
  full_name='syft.grid.service.CloseConnectionMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_id', full_name='syft.grid.service.CloseConnectionMessage.msg_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='syft.grid.service.CloseConnectionMessage.address', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=1472,
  serialized_end=1575,
)

_REGISTERNEWPEERMESSAGE.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_REGISTERNEWPEERMESSAGE.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_REGISTERNEWPEERMESSAGE.fields_by_name['reply_to'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_PEERSUCCESSFULLYREGISTERED.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_PEERSUCCESSFULLYREGISTERED.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_SIGNALINGANSWERMESSAGE.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_SIGNALINGANSWERMESSAGE.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_SIGNALINGANSWERMESSAGE.fields_by_name['host_metadata'].message_type = proto_dot_core_dot_node_dot_common_dot_metadata__pb2._METADATA
_SIGNALINGOFFERMESSAGE.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_SIGNALINGOFFERMESSAGE.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_SIGNALINGOFFERMESSAGE.fields_by_name['host_metadata'].message_type = proto_dot_core_dot_node_dot_common_dot_metadata__pb2._METADATA
_OFFERPULLREQUESTMESSAGE.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_OFFERPULLREQUESTMESSAGE.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_OFFERPULLREQUESTMESSAGE.fields_by_name['reply_to'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_ANSWERPULLREQUESTMESSAGE.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_ANSWERPULLREQUESTMESSAGE.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_ANSWERPULLREQUESTMESSAGE.fields_by_name['reply_to'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_SIGNALINGREQUESTSNOTFOUND.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_SIGNALINGREQUESTSNOTFOUND.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_INVALIDLOOPBACKREQUEST.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_INVALIDLOOPBACKREQUEST.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_CLOSECONNECTIONMESSAGE.fields_by_name['msg_id'].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_CLOSECONNECTIONMESSAGE.fields_by_name['address'].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
DESCRIPTOR.message_types_by_name['RegisterNewPeerMessage'] = _REGISTERNEWPEERMESSAGE
DESCRIPTOR.message_types_by_name['PeerSuccessfullyRegistered'] = _PEERSUCCESSFULLYREGISTERED
DESCRIPTOR.message_types_by_name['SignalingAnswerMessage'] = _SIGNALINGANSWERMESSAGE
DESCRIPTOR.message_types_by_name['SignalingOfferMessage'] = _SIGNALINGOFFERMESSAGE
DESCRIPTOR.message_types_by_name['OfferPullRequestMessage'] = _OFFERPULLREQUESTMESSAGE
DESCRIPTOR.message_types_by_name['AnswerPullRequestMessage'] = _ANSWERPULLREQUESTMESSAGE
DESCRIPTOR.message_types_by_name['SignalingRequestsNotFound'] = _SIGNALINGREQUESTSNOTFOUND
DESCRIPTOR.message_types_by_name['InvalidLoopBackRequest'] = _INVALIDLOOPBACKREQUEST
DESCRIPTOR.message_types_by_name['CloseConnectionMessage'] = _CLOSECONNECTIONMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegisterNewPeerMessage = _reflection.GeneratedProtocolMessageType('RegisterNewPeerMessage', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERNEWPEERMESSAGE,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.RegisterNewPeerMessage)
  })
_sym_db.RegisterMessage(RegisterNewPeerMessage)

PeerSuccessfullyRegistered = _reflection.GeneratedProtocolMessageType('PeerSuccessfullyRegistered', (_message.Message,), {
  'DESCRIPTOR' : _PEERSUCCESSFULLYREGISTERED,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.PeerSuccessfullyRegistered)
  })
_sym_db.RegisterMessage(PeerSuccessfullyRegistered)

SignalingAnswerMessage = _reflection.GeneratedProtocolMessageType('SignalingAnswerMessage', (_message.Message,), {
  'DESCRIPTOR' : _SIGNALINGANSWERMESSAGE,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.SignalingAnswerMessage)
  })
_sym_db.RegisterMessage(SignalingAnswerMessage)

SignalingOfferMessage = _reflection.GeneratedProtocolMessageType('SignalingOfferMessage', (_message.Message,), {
  'DESCRIPTOR' : _SIGNALINGOFFERMESSAGE,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.SignalingOfferMessage)
  })
_sym_db.RegisterMessage(SignalingOfferMessage)

OfferPullRequestMessage = _reflection.GeneratedProtocolMessageType('OfferPullRequestMessage', (_message.Message,), {
  'DESCRIPTOR' : _OFFERPULLREQUESTMESSAGE,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.OfferPullRequestMessage)
  })
_sym_db.RegisterMessage(OfferPullRequestMessage)

AnswerPullRequestMessage = _reflection.GeneratedProtocolMessageType('AnswerPullRequestMessage', (_message.Message,), {
  'DESCRIPTOR' : _ANSWERPULLREQUESTMESSAGE,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.AnswerPullRequestMessage)
  })
_sym_db.RegisterMessage(AnswerPullRequestMessage)

SignalingRequestsNotFound = _reflection.GeneratedProtocolMessageType('SignalingRequestsNotFound', (_message.Message,), {
  'DESCRIPTOR' : _SIGNALINGREQUESTSNOTFOUND,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.SignalingRequestsNotFound)
  })
_sym_db.RegisterMessage(SignalingRequestsNotFound)

InvalidLoopBackRequest = _reflection.GeneratedProtocolMessageType('InvalidLoopBackRequest', (_message.Message,), {
  'DESCRIPTOR' : _INVALIDLOOPBACKREQUEST,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.InvalidLoopBackRequest)
  })
_sym_db.RegisterMessage(InvalidLoopBackRequest)

CloseConnectionMessage = _reflection.GeneratedProtocolMessageType('CloseConnectionMessage', (_message.Message,), {
  'DESCRIPTOR' : _CLOSECONNECTIONMESSAGE,
  '__module__' : 'proto.grid.service.signaling_service_pb2'
  # @@protoc_insertion_point(class_scope:syft.grid.service.CloseConnectionMessage)
  })
_sym_db.RegisterMessage(CloseConnectionMessage)


# @@protoc_insertion_point(module_scope)
