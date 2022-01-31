# stdlib
from typing import Any
from typing import List


class DomainServiceRegistry:
    """A class for registering the services attached to the domain client."""

    __domain_registry: List = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__domain_registry.append(cls)

    @classmethod
    def get_registered_services(cls) -> List[Any]:
        """dict: Returns map of service inheriting this class."""
        return list(cls.__domain_registry)


class NetworkServiceRegistry:
    """A class for registering the services attached to the network client."""

    __slots___ = ("get_registered_services",)

    __network_registry: List = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__network_registry.append(cls)

    @classmethod
    def get_registered_services(cls) -> List[Any]:
        """dict: Returns map of service inheriting this class."""
        return list(cls.__network_registry)
