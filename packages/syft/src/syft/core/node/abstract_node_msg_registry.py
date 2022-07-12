# stdlib
from abc import ABC
from typing import Any
from typing import List


class AbstractNodeMessageRegistry(ABC):
    """A class for registering messages that can be used by a node service."""

    __node_message_registry: List = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__node_message_registry.append(cls)

    @classmethod
    def get_registered_messages(cls) -> List[Any]:
        """dict: Returns map of service inheriting this class."""
        return list(cls.__node_message_registry)
