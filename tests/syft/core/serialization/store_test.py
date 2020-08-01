from syft.core.serialization import serialization_store
from syft.core.common.uid import UID
from syft.core.common.serializable import Serializable


class TestClass(Serializable):
    @staticmethod
    def to_protobuf(self):
        pass

    @staticmethod
    def from_protobuf(proto):
        pass

    @staticmethod
    def get_protobuf_schema():
        pass


def test_lazy_dict():
    serialization_store.clear()
    assert len(serialization_store.schema_to_type) == 0
    assert UID in serialization_store.type_to_schema
    assert len(serialization_store.schema_to_type) != 0


def test_lazy_set_test():
    serialization_store.serde_types.clear()
    assert len(serialization_store) == 0
    assert UID in serialization_store.serde_types
    assert len(serialization_store) != 0


def test_dynamic_test():
    assert TestClass in serialization_store.serde_types
    delattr(TestClass, "to_protobuf")
    serialization_store.clear()
    assert TestClass not in serialization_store.serde_types
