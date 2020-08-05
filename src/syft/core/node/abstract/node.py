from typing import Any, List, Optional, Union

from ...io.location import Location
from ...io.address import Address
from ...store import ObjectStore
from ...common.uid import UID
from ...common.message import (
    ImmediateSyftMessageWithoutReply,
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)


class AbstractNode(Location):
    store: ObjectStore
    lib_ast: Any  # Cant import Globals (circular reference)
    address: Address

    # QUESTION: These are incompatible with the LocationAwareObject properties
    network_id: Optional[Union[str, UID]]
    domain_id: Optional[Union[str, UID]]
    device_id: Optional[Union[str, UID]]
    """"""

    def __init__(self):
        super().__init__()

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

    def recv_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError


class AbstractNodeClient:
    lib_ast: Any  # Cant import Globals (circular reference)
    address: Address
    """"""

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError
