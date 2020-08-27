# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/store/store_object.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/store/store_object.proto",
    package="syft.core.store",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n#proto/core/store/store_object.proto\x12\x0fsyft.core.store\x1a%proto/core/common/common_object.proto\x1a\x19google/protobuf/any.proto"\x82\x01\n\x12StoreGenericObject\x12\x38\n\x0fstorable_object\x18\x01 \x01(\x0b\x32\x1f.syft.core.store.StorableObject\x12\x32\n\x0cstore_object\x18\x02 \x01(\x0b\x32\x1c.syft.core.store.StoreObject"\xa8\x01\n\x0eStorableObject\x12!\n\x02id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x10\n\x08obj_type\x18\x02 \x01(\t\x12\x1a\n\x12schematic_qualname\x18\x03 \x01(\t\x12"\n\x04\x64\x61ta\x18\x04 \x01(\x0b\x32\x14.google.protobuf.Any\x12\x13\n\x0b\x64\x65scription\x18\x05 \x01(\t\x12\x0c\n\x04tags\x18\x06 \x03(\t"r\n\x0bStoreObject\x12-\n\tcontainer\x18\x01 \x01(\x0b\x32\x1a.syft.core.store.Container\x12\x34\n\rsearch_engine\x18\x02 \x01(\x0b\x32\x1d.syft.core.store.SearchEngine"\xb1\x01\n\tContainer\x12\x36\n\x0e\x64ict_container\x18\x01 \x01(\x0b\x32\x1e.syft.core.store.DictContainer\x12\x34\n\rsql_container\x18\x02 \x01(\x0b\x32\x1d.syft.core.store.SqlContainer\x12\x36\n\x0egrid_container\x18\x03 \x01(\x0b\x32\x1e.syft.core.store.GridContainer"\x0e\n\x0cSearchEngine"\xa0\x01\n\rDictContainer\x12\x45\n\x0c\x64ict_mapping\x18\x01 \x03(\x0b\x32/.syft.core.store.DictContainer.DictMappingEntry\x1aH\n\x10\x44ictMappingEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01"\x0e\n\x0cSqlContainer"\x0f\n\rGridContainerb\x06proto3',
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
        google_dot_protobuf_dot_any__pb2.DESCRIPTOR,
    ],
)


_STOREGENERICOBJECT = _descriptor.Descriptor(
    name="StoreGenericObject",
    full_name="syft.core.store.StoreGenericObject",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="storable_object",
            full_name="syft.core.store.StoreGenericObject.storable_object",
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="store_object",
            full_name="syft.core.store.StoreGenericObject.store_object",
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
            create_key=_descriptor._internal_create_key,
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
    serialized_start=123,
    serialized_end=253,
)


_STORABLEOBJECT = _descriptor.Descriptor(
    name="StorableObject",
    full_name="syft.core.store.StorableObject",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="id",
            full_name="syft.core.store.StorableObject.id",
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="obj_type",
            full_name="syft.core.store.StorableObject.obj_type",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="schematic_qualname",
            full_name="syft.core.store.StorableObject.schematic_qualname",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="data",
            full_name="syft.core.store.StorableObject.data",
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="description",
            full_name="syft.core.store.StorableObject.description",
            index=4,
            number=5,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="tags",
            full_name="syft.core.store.StorableObject.tags",
            index=5,
            number=6,
            type=9,
            cpp_type=9,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
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
    serialized_start=256,
    serialized_end=424,
)


_STOREOBJECT = _descriptor.Descriptor(
    name="StoreObject",
    full_name="syft.core.store.StoreObject",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="container",
            full_name="syft.core.store.StoreObject.container",
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="search_engine",
            full_name="syft.core.store.StoreObject.search_engine",
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
            create_key=_descriptor._internal_create_key,
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
    serialized_start=426,
    serialized_end=540,
)


_CONTAINER = _descriptor.Descriptor(
    name="Container",
    full_name="syft.core.store.Container",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="dict_container",
            full_name="syft.core.store.Container.dict_container",
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="sql_container",
            full_name="syft.core.store.Container.sql_container",
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="grid_container",
            full_name="syft.core.store.Container.grid_container",
            index=2,
            number=3,
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
            create_key=_descriptor._internal_create_key,
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
    serialized_start=543,
    serialized_end=720,
)


_SEARCHENGINE = _descriptor.Descriptor(
    name="SearchEngine",
    full_name="syft.core.store.SearchEngine",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=722,
    serialized_end=736,
)


_DICTCONTAINER_DICTMAPPINGENTRY = _descriptor.Descriptor(
    name="DictMappingEntry",
    full_name="syft.core.store.DictContainer.DictMappingEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="syft.core.store.DictContainer.DictMappingEntry.key",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="syft.core.store.DictContainer.DictMappingEntry.value",
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
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=b"8\001",
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=827,
    serialized_end=899,
)

_DICTCONTAINER = _descriptor.Descriptor(
    name="DictContainer",
    full_name="syft.core.store.DictContainer",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="dict_mapping",
            full_name="syft.core.store.DictContainer.dict_mapping",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _DICTCONTAINER_DICTMAPPINGENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=739,
    serialized_end=899,
)


_SQLCONTAINER = _descriptor.Descriptor(
    name="SqlContainer",
    full_name="syft.core.store.SqlContainer",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=901,
    serialized_end=915,
)


_GRIDCONTAINER = _descriptor.Descriptor(
    name="GridContainer",
    full_name="syft.core.store.GridContainer",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=917,
    serialized_end=932,
)

_STOREGENERICOBJECT.fields_by_name["storable_object"].message_type = _STORABLEOBJECT
_STOREGENERICOBJECT.fields_by_name["store_object"].message_type = _STOREOBJECT
_STORABLEOBJECT.fields_by_name[
    "id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_STORABLEOBJECT.fields_by_name[
    "data"
].message_type = google_dot_protobuf_dot_any__pb2._ANY
_STOREOBJECT.fields_by_name["container"].message_type = _CONTAINER
_STOREOBJECT.fields_by_name["search_engine"].message_type = _SEARCHENGINE
_CONTAINER.fields_by_name["dict_container"].message_type = _DICTCONTAINER
_CONTAINER.fields_by_name["sql_container"].message_type = _SQLCONTAINER
_CONTAINER.fields_by_name["grid_container"].message_type = _GRIDCONTAINER
_DICTCONTAINER_DICTMAPPINGENTRY.fields_by_name[
    "value"
].message_type = google_dot_protobuf_dot_any__pb2._ANY
_DICTCONTAINER_DICTMAPPINGENTRY.containing_type = _DICTCONTAINER
_DICTCONTAINER.fields_by_name[
    "dict_mapping"
].message_type = _DICTCONTAINER_DICTMAPPINGENTRY
DESCRIPTOR.message_types_by_name["StoreGenericObject"] = _STOREGENERICOBJECT
DESCRIPTOR.message_types_by_name["StorableObject"] = _STORABLEOBJECT
DESCRIPTOR.message_types_by_name["StoreObject"] = _STOREOBJECT
DESCRIPTOR.message_types_by_name["Container"] = _CONTAINER
DESCRIPTOR.message_types_by_name["SearchEngine"] = _SEARCHENGINE
DESCRIPTOR.message_types_by_name["DictContainer"] = _DICTCONTAINER
DESCRIPTOR.message_types_by_name["SqlContainer"] = _SQLCONTAINER
DESCRIPTOR.message_types_by_name["GridContainer"] = _GRIDCONTAINER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StoreGenericObject = _reflection.GeneratedProtocolMessageType(
    "StoreGenericObject",
    (_message.Message,),
    {
        "DESCRIPTOR": _STOREGENERICOBJECT,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.StoreGenericObject)
    },
)
_sym_db.RegisterMessage(StoreGenericObject)

StorableObject = _reflection.GeneratedProtocolMessageType(
    "StorableObject",
    (_message.Message,),
    {
        "DESCRIPTOR": _STORABLEOBJECT,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.StorableObject)
    },
)
_sym_db.RegisterMessage(StorableObject)

StoreObject = _reflection.GeneratedProtocolMessageType(
    "StoreObject",
    (_message.Message,),
    {
        "DESCRIPTOR": _STOREOBJECT,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.StoreObject)
    },
)
_sym_db.RegisterMessage(StoreObject)

Container = _reflection.GeneratedProtocolMessageType(
    "Container",
    (_message.Message,),
    {
        "DESCRIPTOR": _CONTAINER,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.Container)
    },
)
_sym_db.RegisterMessage(Container)

SearchEngine = _reflection.GeneratedProtocolMessageType(
    "SearchEngine",
    (_message.Message,),
    {
        "DESCRIPTOR": _SEARCHENGINE,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.SearchEngine)
    },
)
_sym_db.RegisterMessage(SearchEngine)

DictContainer = _reflection.GeneratedProtocolMessageType(
    "DictContainer",
    (_message.Message,),
    {
        "DictMappingEntry": _reflection.GeneratedProtocolMessageType(
            "DictMappingEntry",
            (_message.Message,),
            {
                "DESCRIPTOR": _DICTCONTAINER_DICTMAPPINGENTRY,
                "__module__": "proto.core.store.store_object_pb2"
                # @@protoc_insertion_point(class_scope:syft.core.store.DictContainer.DictMappingEntry)
            },
        ),
        "DESCRIPTOR": _DICTCONTAINER,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.DictContainer)
    },
)
_sym_db.RegisterMessage(DictContainer)
_sym_db.RegisterMessage(DictContainer.DictMappingEntry)

SqlContainer = _reflection.GeneratedProtocolMessageType(
    "SqlContainer",
    (_message.Message,),
    {
        "DESCRIPTOR": _SQLCONTAINER,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.SqlContainer)
    },
)
_sym_db.RegisterMessage(SqlContainer)

GridContainer = _reflection.GeneratedProtocolMessageType(
    "GridContainer",
    (_message.Message,),
    {
        "DESCRIPTOR": _GRIDCONTAINER,
        "__module__": "proto.core.store.store_object_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.store.GridContainer)
    },
)
_sym_db.RegisterMessage(GridContainer)


_DICTCONTAINER_DICTMAPPINGENTRY._options = None
# @@protoc_insertion_point(module_scope)
