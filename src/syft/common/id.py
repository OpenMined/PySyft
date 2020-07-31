import uuid
from typing import final
from ..proto import ProtoUID


class AbstractUID:
    ""


@final
class UID(AbstractUID):
    """A unique id"""

    def __init__(self, value=None):
        if value is None:
            value = uuid.uuid4()
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, UID):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"<UID:{self.value}>"

    def serialize(self):
        return ProtoUID(value=self.value.bytes)

    @staticmethod
    def deserialize(proto_uid: ProtoUID) -> AbstractUID:
        return UID(value=uuid.UUID(bytes=proto_uid.value))
