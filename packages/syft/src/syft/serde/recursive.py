# stdlib
from collections.abc import Callable
from enum import Enum
from enum import EnumMeta
import os
import tempfile
import types
from typing import Any

# third party
from capnp.lib.capnp import _DynamicStructBuilder
from pydantic import BaseModel

# syft absolute
import syft as sy

# relative
from ..types.syft_object_registry import SyftObjectRegistry
from .capnp import get_capnp_schema
from .util import compatible_with_large_file_writes_capnp

TYPE_BANK = {}  # type: ignore
SYFT_CLASSES_MISSING_CANONICAL_NAME = []

recursive_scheme = get_capnp_schema("recursive_serde.capnp").RecursiveSerde

SPOOLED_FILE_MAX_SIZE_SERDE = 50 * (1024**2)  # 50MB
DEFAULT_EXCLUDE_ATTRS: set[str] = {"syft_pre_hooks__", "syft_post_hooks__"}


def get_types(cls: type, keys: list[str] | None = None) -> list[type] | None:
    if keys is None:
        return None
    types = []
    for key in keys:
        _type = None
        annotations = getattr(cls, "__annotations__", None)
        if annotations and key in annotations:
            _type = annotations[key]
        else:
            for parent_cls in cls.mro():
                sub_annotations = getattr(parent_cls, "__annotations__", None)
                if sub_annotations and key in sub_annotations:
                    _type = sub_annotations[key]
        if _type is None:
            return None
        types.append(_type)
    return types


def check_fqn_alias(cls: object | type) -> tuple[str, ...] | None:
    """Currently, typing.Any has different metaclasses in different versions of Python 🤦‍♂️.
    For Python <=3.10
    Any is an instance of typing._SpecialForm

    For Python >=3.11
    Any is an instance of typing._AnyMeta
    Hence adding both the aliases to the type bank.

    This would cause issues, when the server and client
    have different python versions.

    As their serde is same, we can use the same serde for both of them.
    with aliases for  fully qualified names in type bank

    In a similar manner for Enum.

    For Python<=3.10:
    Enum is metaclass of enum.EnumMeta

    For Python>=3.11:
    Enum is metaclass of enum.EnumType
    """
    if cls == Any:
        return ("typing._AnyMeta", "typing._SpecialForm")
    if cls == EnumMeta:
        return ("enum.EnumMeta", "enum.EnumType")

    return None


def has_canonical_name_version(
    cls: type, cannonical_name: str | None, version: int | None
) -> bool:
    cls_canonical_name = getattr(cls, "__canonical_name__", None)
    cls_version = getattr(cls, "__version__", None)
    return bool(cls_canonical_name or cannonical_name) and bool(cls_version or version)


def validate_cannonical_name_version(
    cls: type, canonical_name: str | None, version: int | None
) -> tuple[str, int]:
    cls_canonical_name = getattr(cls, "__canonical_name__", None)
    cls_version = getattr(cls, "__version__", None)
    if cls_canonical_name and canonical_name:
        raise ValueError(
            "Cannot specify both __canonical_name__ attribute and cannonical_name argument."
        )
    if cls_version and version:
        raise ValueError(
            "Cannot specify both __version__ attribute and version argument."
        )
    if cls_canonical_name is None and canonical_name is None:
        raise ValueError(
            "Must specify either __canonical_name__ attribute or cannonical_name argument."
        )
    if cls_version is None and version is None:
        raise ValueError(
            "Must specify either __version__ attribute or version argument."
        )

    canonical_name = canonical_name or cls_canonical_name
    version = version or cls_version
    return canonical_name, version  # type: ignore


def skip_unregistered_class(
    cls: type, canonical_name: str | None, version: str | None
) -> bool:
    """
    Used to gather all classes that are missing canonical_name and version for development.

    Returns True if the class should be skipped, False otherwise.
    """

    search_unregistered_classes = (
        os.getenv("SYFT_SEARCH_MISSING_CANONICAL_NAME", False) == "true"
    )
    if not search_unregistered_classes:
        return False
    if not has_canonical_name_version(cls, canonical_name, version):
        if cls.__module__.startswith("syft."):
            SYFT_CLASSES_MISSING_CANONICAL_NAME.append(cls)
            return True
    return False


def recursive_serde_register(
    cls: object | type,
    serialize: Callable | None = None,
    deserialize: Callable | None = None,
    serialize_attrs: list | None = None,
    exclude_attrs: list | None = None,
    inherit_attrs: bool | None = True,
    inheritable_attrs: bool | None = True,
    canonical_name: str | None = None,
    version: int | None = None,
) -> None:
    pydantic_fields = None
    base_attrs = None
    attribute_list: set[str] = set()

    cls = type(cls) if not isinstance(cls, type) else cls

    if skip_unregistered_class(cls, canonical_name, version):
        return

    canonical_name, version = validate_cannonical_name_version(
        cls, canonical_name, version
    )

    nonrecursive = bool(serialize and deserialize)
    _serialize = serialize if nonrecursive else rs_object2proto
    _deserialize = deserialize if nonrecursive else rs_proto2object
    is_pydantic = issubclass(cls, BaseModel)
    hash_exclude_attrs = getattr(cls, "__hash_exclude_attrs__", [])

    if inherit_attrs and not is_pydantic:
        # get attrs from base class
        base_attrs = getattr(cls, "__syft_serializable__", [])
        attribute_list.update(base_attrs)

    if is_pydantic and not serialize_attrs:
        # if pydantic object and attrs are provided, the get attrs from __fields__
        # cls.__fields__ auto inherits attrs
        pydantic_fields = [
            field
            for field, field_info in cls.model_fields.items()
            if not (
                field_info.annotation is not None
                and hasattr(field_info.annotation, "__origin__")
                and field_info.annotation.__origin__
                in (Callable, types.FunctionType, types.LambdaType)
            )
        ]
        attribute_list.update(pydantic_fields)

    if serialize_attrs:
        # If serialize_attrs is provided, append it to our attr list
        attribute_list.update(serialize_attrs)

    if issubclass(cls, Enum):
        attribute_list.update(["value"])

    exclude_attrs = [] if exclude_attrs is None else exclude_attrs
    attribute_list = attribute_list - set(exclude_attrs) - DEFAULT_EXCLUDE_ATTRS

    if inheritable_attrs and attribute_list and not is_pydantic:
        # only set __syft_serializable__ for non-pydantic classes because
        # pydantic objects inherit by default
        cls.__syft_serializable__ = attribute_list

    attributes = set(attribute_list) if attribute_list else None
    attribute_types = get_types(cls, attributes)
    serde_overrides = getattr(cls, "__serde_overrides__", {})

    # without fqn duplicate class names overwrite
    serde_attributes = (
        nonrecursive,
        _serialize,
        _deserialize,
        attributes,
        exclude_attrs,
        serde_overrides,
        hash_exclude_attrs,
        cls,
        attribute_types,
        version,
    )

    SyftObjectRegistry.register_cls(canonical_name, version, serde_attributes)

    alias_fqn = check_fqn_alias(cls)
    if isinstance(alias_fqn, tuple):
        for alias in alias_fqn:
            alias_canonical_name = canonical_name + f"_{alias}"
            SyftObjectRegistry.register_cls(alias_canonical_name, 1, serde_attributes)


def chunk_bytes(
    field_obj: Any,
    ser_func: Callable,
    field_name: str | int,
    builder: _DynamicStructBuilder,
) -> None:
    data = ser_func(field_obj)
    size_of_data = len(data)
    if compatible_with_large_file_writes_capnp(size_of_data):
        with tempfile.TemporaryFile() as tmp_file:
            # Write data to a file to save RAM
            tmp_file.write(data)
            tmp_file.seek(0)
            del data

            CHUNK_SIZE = int(5.12e8)  # capnp max for a List(Data) field
            list_size = size_of_data // CHUNK_SIZE + 1
            data_lst = builder.init(field_name, list_size)
            for idx in range(list_size):
                bytes_to_read = min(CHUNK_SIZE, size_of_data)
                data_lst[idx] = tmp_file.read(bytes_to_read)
                size_of_data -= CHUNK_SIZE
    else:
        CHUNK_SIZE = int(5.12e8)  # capnp max for a List(Data) field
        list_size = len(data) // CHUNK_SIZE + 1
        data_lst = builder.init(field_name, list_size)
        END_INDEX = CHUNK_SIZE
        for idx in range(list_size):
            START_INDEX = idx * CHUNK_SIZE
            END_INDEX = min(START_INDEX + CHUNK_SIZE, len(data))
            data_lst[idx] = data[START_INDEX:END_INDEX]


def combine_bytes(capnp_list: list[bytes]) -> bytes:
    # TODO: make sure this doesn't copy, perhaps allocate a fixed size buffer
    # and move the bytes into it as we go
    bytes_value = b""
    for value in capnp_list:
        bytes_value += value
    return bytes_value


def rs_object2proto(self: Any, for_hashing: bool = False) -> _DynamicStructBuilder:
    # relative
    from ..types.syft_object import DYNAMIC_SYFT_ATTRIBUTES

    is_type = False
    if isinstance(self, type):
        is_type = True

    msg = recursive_scheme.new_message()

    # todo: rewrite and make sure every object has a canonical name and version
    canonical_name, version = SyftObjectRegistry.get_canonical_name_version(self)

    if not SyftObjectRegistry.has_serde_class(canonical_name, version):
        # third party
        raise Exception(
            f"obj2proto: {canonical_name} version {version} not in SyftObjectRegistry"
        )

    msg.canonicalName = canonical_name
    msg.version = version

    (
        nonrecursive,
        serialize,
        _,
        attribute_list,
        exclude_attrs_list,
        serde_overrides,
        hash_exclude_attrs,
        _,
        _,
        _,
    ) = SyftObjectRegistry.get_serde_properties(canonical_name, version)

    if nonrecursive or is_type:
        if serialize is None:
            raise Exception(
                f"Cant serialize {type(self)} nonrecursive without serialize."
            )
        chunk_bytes(self, serialize, "nonrecursiveBlob", msg)
        return msg

    if attribute_list is None:
        attribute_list = self.__dict__.keys()

    hash_exclude_attrs_set = (
        set(hash_exclude_attrs).union(set(DYNAMIC_SYFT_ATTRIBUTES))
        if for_hashing
        else set()
    )
    attribute_list = (
        set(attribute_list) - set(exclude_attrs_list) - hash_exclude_attrs_set
    )

    msg.init("fieldsName", len(attribute_list))
    msg.init("fieldsData", len(attribute_list))

    for idx, attr_name in enumerate(sorted(attribute_list)):
        if not hasattr(self, attr_name):
            raise ValueError(
                f"{attr_name} on {type(self)} does not exist, serialization aborted!"
            )

        field_obj = getattr(self, attr_name)
        transforms = serde_overrides.get(attr_name, None)

        if transforms is not None:
            field_obj = transforms[0](field_obj)

        if isinstance(field_obj, types.FunctionType):
            continue

        msg.fieldsName[idx] = attr_name
        chunk_bytes(
            field_obj,
            lambda x: sy.serialize(x, to_bytes=True, for_hashing=for_hashing),
            idx,
            msg.fieldsData,
        )

    return msg


def rs_bytes2object(blob: bytes) -> Any:
    MAX_TRAVERSAL_LIMIT = 2**64 - 1

    with recursive_scheme.from_bytes(
        blob, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    ) as msg:
        return rs_proto2object(msg)


def map_fqns_for_backward_compatibility(fqn: str) -> str:
    """for backwards compatibility with 0.8.6. Sometimes classes where moved to another file. Which is
    exactly why we are implementing it differently"""
    mapping = {
        "syft.service.dataset.dataset.MarkdownDescription": "syft.util.misc_objs.MarkdownDescription",
        # "syft.service.object_search.object_migration_state.SyftObjectMigrationState": "syft.service.migration.object_migration_state.SyftObjectMigrationState",  # noqa: E501
    }
    if fqn in mapping:
        return mapping[fqn]
    else:
        return fqn


def rs_proto2object(proto: _DynamicStructBuilder) -> Any:
    # relative
    from .deserialize import _deserialize

    class_type: type | Any = type(None)

    canonical_name = proto.canonicalName
    version = getattr(proto, "version", -1)

    if not SyftObjectRegistry.has_serde_class(canonical_name, version):
        # relative
        from ..server.server import CODE_RELOADER

        for load_user_code in CODE_RELOADER.values():
            load_user_code()
        # third party
        if not SyftObjectRegistry.has_serde_class(canonical_name, version):
            raise Exception(
                f"proto2obj: {canonical_name} version {version} not in SyftObjectRegistry"
            )

    # TODO: 🐉 sort this out, basically sometimes the syft.user classes are not in the
    # module name space in sub-processes or threads even though they are loaded on start
    # its possible that the uvicorn awsgi server is preloading a bunch of threads
    # however simply getting the class from the TYPE_BANK doesn't always work and
    # causes some errors so it seems like we want to get the local one where possible
    (
        nonrecursive,
        _,
        deserialize,
        _,
        _,
        serde_overrides,
        _,
        cls,
        _,
        version,
    ) = SyftObjectRegistry.get_serde_properties(canonical_name, version)

    class_type = cls

    if nonrecursive:
        if deserialize is None:
            raise Exception(
                f"Cant serialize {type(proto)} nonrecursive without serialize."
            )

        return deserialize(combine_bytes(proto.nonrecursiveBlob))

    kwargs = {}

    for attr_name, attr_bytes_list in zip(proto.fieldsName, proto.fieldsData):
        if attr_name != "":
            attr_bytes = combine_bytes(attr_bytes_list)
            attr_value = _deserialize(attr_bytes, from_bytes=True)
            transforms = serde_overrides.get(attr_name, None)

            if transforms is not None:
                attr_value = transforms[1](attr_value)
            kwargs[attr_name] = attr_value

    if hasattr(class_type, "serde_constructor"):
        return class_type.serde_constructor(kwargs)

    if issubclass(class_type, Enum) and "value" in kwargs:
        obj = class_type.__new__(class_type, kwargs["value"])
    elif issubclass(class_type, BaseModel):
        # if we skip the __new__ flow of BaseModel we get the error
        # AttributeError: object has no attribute '__fields_set__'

        # if "syft.user" in proto.fullyQualifiedName:
        #     # weird issues with pydantic and ForwardRef on user classes being inited
        #     # with custom state args / kwargs
        #     obj = class_type()
        #     for attr_name, attr_value in kwargs.items():
        #         setattr(obj, attr_name, attr_value)
        # else:
        #     obj = class_type(**kwargs)
        obj = class_type(**kwargs)

    else:
        obj = class_type.__new__(class_type)  # type: ignore
        for attr_name, attr_value in kwargs.items():
            setattr(obj, attr_name, attr_value)

    return obj


# how else do you import a relative file to execute it?
NOTHING = None
