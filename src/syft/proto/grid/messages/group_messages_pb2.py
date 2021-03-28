# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/grid/messages/group_messages.proto

# stdlib
import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
# third party
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# syft absolute
from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/grid/messages/group_messages.proto",
    package="syft.grid.messages",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n(proto/grid/messages/group_messages.proto\x12\x12syft.grid.messages\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto"\x9d\x01\n\x12\x43reateGroupMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x8a\x01\n\x13\x43reateGroupResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9a\x01\n\x0fGetGroupMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x87\x01\n\x10GetGroupResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9b\x01\n\x10GetGroupsMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x88\x01\n\x11GetGroupsResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9d\x01\n\x12\x44\x65leteGroupMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x8a\x01\n\x13\x44\x65leteGroupResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9d\x01\n\x12UpdateGroupMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x8a\x01\n\x13UpdateGroupResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Addressb\x06proto3'
    ),
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
        proto_dot_core_dot_io_dot_address__pb2.DESCRIPTOR,
    ],
)


_CREATEGROUPMESSAGE = _descriptor.Descriptor(
    name="CreateGroupMessage",
    full_name="syft.grid.messages.CreateGroupMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.CreateGroupMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.CreateGroupMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.CreateGroupMessage.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="reply_to",
            full_name="syft.grid.messages.CreateGroupMessage.reply_to",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=133,
    serialized_end=290,
)


_CREATEGROUPRESPONSE = _descriptor.Descriptor(
    name="CreateGroupResponse",
    full_name="syft.grid.messages.CreateGroupResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.CreateGroupResponse.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="status_code",
            full_name="syft.grid.messages.CreateGroupResponse.status_code",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.CreateGroupResponse.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.CreateGroupResponse.address",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=293,
    serialized_end=431,
)


_GETGROUPMESSAGE = _descriptor.Descriptor(
    name="GetGroupMessage",
    full_name="syft.grid.messages.GetGroupMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.GetGroupMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.GetGroupMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.GetGroupMessage.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="reply_to",
            full_name="syft.grid.messages.GetGroupMessage.reply_to",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=434,
    serialized_end=588,
)


_GETGROUPRESPONSE = _descriptor.Descriptor(
    name="GetGroupResponse",
    full_name="syft.grid.messages.GetGroupResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.GetGroupResponse.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="status_code",
            full_name="syft.grid.messages.GetGroupResponse.status_code",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.GetGroupResponse.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.GetGroupResponse.address",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=591,
    serialized_end=726,
)


_GETGROUPSMESSAGE = _descriptor.Descriptor(
    name="GetGroupsMessage",
    full_name="syft.grid.messages.GetGroupsMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.GetGroupsMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.GetGroupsMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.GetGroupsMessage.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="reply_to",
            full_name="syft.grid.messages.GetGroupsMessage.reply_to",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=729,
    serialized_end=884,
)


_GETGROUPSRESPONSE = _descriptor.Descriptor(
    name="GetGroupsResponse",
    full_name="syft.grid.messages.GetGroupsResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.GetGroupsResponse.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="status_code",
            full_name="syft.grid.messages.GetGroupsResponse.status_code",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.GetGroupsResponse.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.GetGroupsResponse.address",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=887,
    serialized_end=1023,
)


_DELETEGROUPMESSAGE = _descriptor.Descriptor(
    name="DeleteGroupMessage",
    full_name="syft.grid.messages.DeleteGroupMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.DeleteGroupMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.DeleteGroupMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.DeleteGroupMessage.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="reply_to",
            full_name="syft.grid.messages.DeleteGroupMessage.reply_to",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=1026,
    serialized_end=1183,
)


_DELETEGROUPRESPONSE = _descriptor.Descriptor(
    name="DeleteGroupResponse",
    full_name="syft.grid.messages.DeleteGroupResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.DeleteGroupResponse.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="status_code",
            full_name="syft.grid.messages.DeleteGroupResponse.status_code",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.DeleteGroupResponse.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.DeleteGroupResponse.address",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=1186,
    serialized_end=1324,
)


_UPDATEGROUPMESSAGE = _descriptor.Descriptor(
    name="UpdateGroupMessage",
    full_name="syft.grid.messages.UpdateGroupMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.UpdateGroupMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.UpdateGroupMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.UpdateGroupMessage.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="reply_to",
            full_name="syft.grid.messages.UpdateGroupMessage.reply_to",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=1327,
    serialized_end=1484,
)


_UPDATEGROUPRESPONSE = _descriptor.Descriptor(
    name="UpdateGroupResponse",
    full_name="syft.grid.messages.UpdateGroupResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.grid.messages.UpdateGroupResponse.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="status_code",
            full_name="syft.grid.messages.UpdateGroupResponse.status_code",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="content",
            full_name="syft.grid.messages.UpdateGroupResponse.content",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.grid.messages.UpdateGroupResponse.address",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=1487,
    serialized_end=1625,
)

_CREATEGROUPMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_CREATEGROUPMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_CREATEGROUPMESSAGE.fields_by_name[
    "reply_to"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_CREATEGROUPRESPONSE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_CREATEGROUPRESPONSE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_GETGROUPMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_GETGROUPMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_GETGROUPMESSAGE.fields_by_name[
    "reply_to"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_GETGROUPRESPONSE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_GETGROUPRESPONSE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_GETGROUPSMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_GETGROUPSMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_GETGROUPSMESSAGE.fields_by_name[
    "reply_to"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_GETGROUPSRESPONSE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_GETGROUPSRESPONSE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_DELETEGROUPMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_DELETEGROUPMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_DELETEGROUPMESSAGE.fields_by_name[
    "reply_to"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_DELETEGROUPRESPONSE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_DELETEGROUPRESPONSE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_UPDATEGROUPMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_UPDATEGROUPMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_UPDATEGROUPMESSAGE.fields_by_name[
    "reply_to"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_UPDATEGROUPRESPONSE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_UPDATEGROUPRESPONSE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
DESCRIPTOR.message_types_by_name["CreateGroupMessage"] = _CREATEGROUPMESSAGE
DESCRIPTOR.message_types_by_name["CreateGroupResponse"] = _CREATEGROUPRESPONSE
DESCRIPTOR.message_types_by_name["GetGroupMessage"] = _GETGROUPMESSAGE
DESCRIPTOR.message_types_by_name["GetGroupResponse"] = _GETGROUPRESPONSE
DESCRIPTOR.message_types_by_name["GetGroupsMessage"] = _GETGROUPSMESSAGE
DESCRIPTOR.message_types_by_name["GetGroupsResponse"] = _GETGROUPSRESPONSE
DESCRIPTOR.message_types_by_name["DeleteGroupMessage"] = _DELETEGROUPMESSAGE
DESCRIPTOR.message_types_by_name["DeleteGroupResponse"] = _DELETEGROUPRESPONSE
DESCRIPTOR.message_types_by_name["UpdateGroupMessage"] = _UPDATEGROUPMESSAGE
DESCRIPTOR.message_types_by_name["UpdateGroupResponse"] = _UPDATEGROUPRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CreateGroupMessage = _reflection.GeneratedProtocolMessageType(
    "CreateGroupMessage",
    (_message.Message,),
    dict(
        DESCRIPTOR=_CREATEGROUPMESSAGE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.CreateGroupMessage)
    ),
)
_sym_db.RegisterMessage(CreateGroupMessage)

CreateGroupResponse = _reflection.GeneratedProtocolMessageType(
    "CreateGroupResponse",
    (_message.Message,),
    dict(
        DESCRIPTOR=_CREATEGROUPRESPONSE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.CreateGroupResponse)
    ),
)
_sym_db.RegisterMessage(CreateGroupResponse)

GetGroupMessage = _reflection.GeneratedProtocolMessageType(
    "GetGroupMessage",
    (_message.Message,),
    dict(
        DESCRIPTOR=_GETGROUPMESSAGE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetGroupMessage)
    ),
)
_sym_db.RegisterMessage(GetGroupMessage)

GetGroupResponse = _reflection.GeneratedProtocolMessageType(
    "GetGroupResponse",
    (_message.Message,),
    dict(
        DESCRIPTOR=_GETGROUPRESPONSE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetGroupResponse)
    ),
)
_sym_db.RegisterMessage(GetGroupResponse)

GetGroupsMessage = _reflection.GeneratedProtocolMessageType(
    "GetGroupsMessage",
    (_message.Message,),
    dict(
        DESCRIPTOR=_GETGROUPSMESSAGE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetGroupsMessage)
    ),
)
_sym_db.RegisterMessage(GetGroupsMessage)

GetGroupsResponse = _reflection.GeneratedProtocolMessageType(
    "GetGroupsResponse",
    (_message.Message,),
    dict(
        DESCRIPTOR=_GETGROUPSRESPONSE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetGroupsResponse)
    ),
)
_sym_db.RegisterMessage(GetGroupsResponse)

DeleteGroupMessage = _reflection.GeneratedProtocolMessageType(
    "DeleteGroupMessage",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DELETEGROUPMESSAGE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.DeleteGroupMessage)
    ),
)
_sym_db.RegisterMessage(DeleteGroupMessage)

DeleteGroupResponse = _reflection.GeneratedProtocolMessageType(
    "DeleteGroupResponse",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DELETEGROUPRESPONSE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.DeleteGroupResponse)
    ),
)
_sym_db.RegisterMessage(DeleteGroupResponse)

UpdateGroupMessage = _reflection.GeneratedProtocolMessageType(
    "UpdateGroupMessage",
    (_message.Message,),
    dict(
        DESCRIPTOR=_UPDATEGROUPMESSAGE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.UpdateGroupMessage)
    ),
)
_sym_db.RegisterMessage(UpdateGroupMessage)

UpdateGroupResponse = _reflection.GeneratedProtocolMessageType(
    "UpdateGroupResponse",
    (_message.Message,),
    dict(
        DESCRIPTOR=_UPDATEGROUPRESPONSE,
        __module__="proto.grid.messages.group_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.UpdateGroupResponse)
    ),
)
_sym_db.RegisterMessage(UpdateGroupResponse)


# @@protoc_insertion_point(module_scope)
