# relative
from .abstract_node_msg_registry import AbstractNodeMessageRegistry


class DomainMessageRegistry(AbstractNodeMessageRegistry):
    """A class for registering messages that can be used by the domain service."""
