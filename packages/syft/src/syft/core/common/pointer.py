# stdlib
from typing import Any
from typing import List
from typing import Optional

# relative
from .uid import UID


class AbstractPointer:
    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
    ):
        if id_at_location is None:
            id_at_location = UID()

        if tags is None:
            tags = []

        self.client = client
        self.id_at_location = id_at_location
        self.tags = tags
        self.description = description
        self.gc_enabled = True

        self.is_enum = False
