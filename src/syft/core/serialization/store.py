from .serializable import get_protobuf_classes, get_protobuf_wrappers
from ...common import LazyDict, LazySet


class SerializationStore:
    """
    Store to cache all serialization methods and different from the type to
    its protobuf schema and the other way around.

    Attributes:
        serde_types (set): all the types that can be used for serde.
        _type_to_schema (LazyDict): mapping from to type of an object to the
        schema that serializes it.
        _schema_to_type (LazyDict): mapping from the type of a schema to the
        type that can be serialized with that schema.
    """

    __slots__ = ["serde_types", "_type_to_schema", "_schema_to_type"]

    def __init__(self):
        self.serde_types = LazySet(self.lazy_update)
        self._type_to_schema = LazyDict(self.lazy_update)
        self._schema_to_type = LazyDict(self.lazy_update)

    @property
    def type_to_schema(self):
        return self._type_to_schema

    @property
    def schema_to_type(self):
        return self._schema_to_type

    def lazy_update(self) -> None:
        """
        Method that updates the content of store when an object is not found.
        It relies on the fact that new items might appear when traversing
        the tree-like structure of the mro of the Serializable interface.
        """
        classes = get_protobuf_classes()
        wrappers = get_protobuf_wrappers()
        for proto_class in classes:
            self.serde_types.add(proto_class)
            self._type_to_schema[proto_class] = proto_class.get_protobuf_schema()
            self._schema_to_type[proto_class.get_protobuf_schema()] = proto_class

        for wrapper in wrappers:
            self.serde_types.add(wrapper)
            self._type_to_schema[
                wrapper.get_protobuf_wrapper()
            ] = wrapper.get_protobuf_schema()
            self._schema_to_type[
                wrapper.get_protobuf_schema()
            ] = wrapper.get_protobuf_wrapper()

    def clear(self):
        self.serde_types.clear()
        self._type_to_schema.clear()
        self._schema_to_type.clear()

    def __len__(self):
        return len(self.serde_types)


serialization_store = SerializationStore()
