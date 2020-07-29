import uuid
from typing import final
from syft.interfaces import Serializable


@final
class UID(Serializable):
    """A unique id"""

    def __init__(self):
        # TODO find out what type is smaller for protobuf msgs.
        self.value = uuid.uuid4()

    @staticmethod
    def to_protobuf(obj):
        pass

    @staticmethod
    def from_protobuf(proto):
        pass

    @staticmethod
    def get_protobuf_schema():
        pass

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, UID):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"<UID:{self.value}>"
