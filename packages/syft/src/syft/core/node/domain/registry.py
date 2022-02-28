# stdlib
from typing import Any
from typing import List


class DomainMessageRegistry:
    """A class for registering messages that can be used by the domain service."""

    __domain_message_registry: List = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__domain_message_registry.append(cls)

    @classmethod
    def get_registered_messages(cls) -> List[Any]:
        """dict: Returns map of service inheriting this class."""
        return list(cls.__domain_message_registry)
