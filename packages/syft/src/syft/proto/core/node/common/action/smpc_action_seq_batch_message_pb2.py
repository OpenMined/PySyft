# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/action/smpc_action_seq_batch_message.proto
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
from syft.proto.core.node.common.action import (
    smpc_action_message_pb2 as proto_dot_core_dot_node_dot_common_dot_action_dot_smpc__action__message__pb2,
)


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\nAproto/core/node/common/action/smpc_action_seq_batch_message.proto\x12\x1csyft.core.node.common.action\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto\x1a\x37proto/core/node/common/action/smpc_action_message.proto"\xb1\x01\n\x19SMPCActionSeqBatchMessage\x12\x45\n\x0csmpc_actions\x18\x01 \x03(\x0b\x32/.syft.core.node.common.action.SMPCActionMessage\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12%\n\x06msg_id\x18\x03 \x01(\x0b\x32\x15.syft.core.common.UIDb\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR,
    "proto.core.node.common.action.smpc_action_seq_batch_message_pb2",
    globals(),
)
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SMPCACTIONSEQBATCHMESSAGE._serialized_start = 225
    _SMPCACTIONSEQBATCHMESSAGE._serialized_end = 402
# @@protoc_insertion_point(module_scope)
