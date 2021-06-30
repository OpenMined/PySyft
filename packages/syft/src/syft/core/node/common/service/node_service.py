# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from .....logger import traceback_and_raise
from ...abstract.node import AbstractNode


class NodeService:
    @staticmethod
    def message_handler_types() -> List[Type[Any]]:
        traceback_and_raise(NotImplementedError)

    @classmethod
    def class_name(cls) -> str:
        return str(cls.__name__)

    @classmethod
    def icon(cls) -> str:
        return "⚙️ "

    @classmethod
    def pprint(cls) -> str:
        return f"{cls.icon()} ({cls.class_name()})"

    @staticmethod
    def process(
        node: AbstractNode, msg: Any, verify_key: Optional[VerifyKey] = None
    ) -> Any:
        traceback_and_raise(NotImplementedError)


class ImmediateNodeService(NodeService):
    """A service for messages which should be immediately executed"""

    pass


class EventualNodeService(NodeService):
    """A service for messages which need not be immediately executed
    but which can be executed at the worker's convenience"""

    pass


class ImmediateNodeServiceWithReply(ImmediateNodeService):
    pass


class EventualNodeServiceWithoutReply(NodeService):
    pass


class ImmediateNodeServiceWithoutReply(NodeService):
    pass


class SignedNodeServiceWithReply(ImmediateNodeService):
    pass


class SignedNodeServiceWithoutReply(ImmediateNodeService):
    pass
