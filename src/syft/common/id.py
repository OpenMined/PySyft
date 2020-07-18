import uuid
from ..decorators import type_hints
from typing import final


@final
class UID(object):
    """A unique id"""

    def __init__(self):
        # TODO find out what type is smaller for protobuf msgs.
        self.value = uuid.uuid4()
