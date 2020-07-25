from dataclasses import dataclass
from .serializable import get_protobuf_classes, get_protobuf_wrappers


@dataclass(frozen=True)
class LazyDict(dict):
    _dict = {}

    def __sizeof__(self):
        return self._dict.__sizeof__()

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def __contains__(self, item) -> bool:
        contains = item in self._dict
        if not contains:
            self.update()
            contains = item in self._dict
        return contains

    def __setitem__(self, key, value) -> None:
        self._dict[key] = value.serialize()

    def __getitem__(self, item):
        if item not in self._dict:
            serialization_store.lazy_update()
        return self._dict[item]


@dataclass(frozen=True)
class SerializationStore:
    serde_types = set()
    _type_to_schema = LazyDict()
    _schema_to_type = LazyDict()

    @property
    def type_to_schema(self):
        return self._type_to_schema

    @property
    def schema_to_type(self):
        return self._schema_to_type

    def lazy_update(self):
        classes = get_protobuf_classes()
        wrappers = get_protobuf_wrappers()


        for proto_class in classes:
            self.serde_types.add(proto_class)
            self._type_to_schema[proto_class] = proto_class.get_protobuf_schema()
            self._schema_to_type[proto_class.get_protobuf_schema()] = proto_class

        for wrapper in wrappers:
            self.serde_types.add(proto_class)
            self._type_to_schema[wrapper.get_protobuf_wrapper()] = wrapper.get_protobuf_schema()
            self._schema_to_type[wrapper.get_protobuf_schema()] = wrapper.get_protobuf_wrapper()



serialization_store = SerializationStore()