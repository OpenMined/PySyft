from typing import Any, List, Optional

from ...common.uid import UID
from ...io.address import Address
from syft.core.io.location import Location
from syft.decorators import syft_decorator
from ...store import ObjectStore
from ...common.message import (
    ImmediateSyftMessageWithoutReply,
    EventualSyftMessageWithoutReply,
    SignedMessage,
)


class AbstractNode(Address):
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):
        super().__init__(network=network, domain=domain, device=device, vm=vm)

    name: Optional[str]
    store: ObjectStore
    lib_ast: Any  # Cant import Globals (circular reference)
    """"""

    @property
    def known_child_nodes(self) -> List[Any]:
        raise NotImplementedError

    def recv_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_immediate_msg_with_reply(self, msg: SignedMessage) -> SignedMessage:
        raise NotImplementedError

    def recv_signed_msg_with_reply(self, msg: SignedMessage) -> SignedMessage:
        raise NotImplementedError

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError


class AbstractNodeClient(Address):
    lib_ast: Any  # Cant import Globals (circular reference)
    address: Address
    """"""

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError
