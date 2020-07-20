from dataclasses import dataclass
from .serializable import get_protobuf_classes, get_protobuf_wrappers


@dataclass(frozen=True)
class LazyDict(dict):
    """
    Struct that simulates the behavior of a normal dictionary, but that a
    fallback update method when an object is not found to update the
    dictionary.

    The overall behavior is:
    * if the element if found, do nothing.
    * else, update the elements of the dicts in a lazy manner.
    * retry the search.

    Attributes:
         _dict: internal dict to store the elements of the lazy dict.

    """
    _dict = {}

    def __sizeof__(self) -> int:
        """
        Method that returns the size of the wrapped dict.

        Returns:
              int: size of the original dict.
        """
        return self._dict.__sizeof__()

    def __len__(self) -> int:
        """
        Method that returns the size of the wrapped dict.

        Returns:
            int: length of the original dict.
        """
        return len(self._dict)

    def keys(self) -> any:
        """
        Method that returns the keys used in the wrapped dict.

        Returns:
            any: the keys used for indexing.
        """
        return self._dict.keys()

    def values(self) -> any:
        """
        Method that returns the values stored in the wrapped dict.

        Returns:
            any: they values stored in the dict.
        """
        return self._dict.values()

    def __contains__(self, item: any) -> bool:
        """
        Method that checks if an object is being used as a key in the wrapped
        dict.

        Args:
            item (any): the key to be searched for.

        Returns:
            bool: if the object is present or not.
        """
        contains = item in self._dict
        if not contains:
            self.update()
            contains = item in self._dict
        return contains

    def __setitem__(self, key: any, value: any) -> None:
        """
        Method that sets an object at a given key in the wrapped dict.

        Args:
              key (any): the key to be used in the dict.
              value (any): the value to be used in the dict.
        """
        self._dict[key] = value.serialize()

    def __getitem__(self, item):
        if item not in self._dict:
            serialization_store.lazy_update()
        return self._dict[item]


@dataclass(frozen=True)
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
    serde_types = set()
    _type_to_schema = LazyDict()
    _schema_to_type = LazyDict()

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
            self._type_to_schema[wrapper.get_protobuf_wrapper()] = wrapper.get_protobuf_schema()
            self._schema_to_type[wrapper.get_protobuf_schema()] = wrapper.get_protobuf_wrapper()


serialization_store = SerializationStore()