import uuid
from typing import final


@final
class UID(object):
    """A unique id"""

    def __init__(self):
        # TODO find out what type is smaller for protobuf msgs.
        self.value = uuid.uuid4()

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, UID):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"<UID:{self.value}>"

