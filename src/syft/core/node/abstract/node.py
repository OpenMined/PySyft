from typing import Any, List, Optional, Union

from ...io.location import Location
from ...io.address import Address, All, Unspecified
from ...store import ObjectStore
from ...common.message import (
    ImmediateSyftMessageWithoutReply,
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)


class AbstractNode(Location):
    store: ObjectStore
    lib_ast: Any  # Cant import Globals (circular reference)

    # QUESTION: How can this match the LocationAwareObject property?
    # Definition of "address" in base class "AbstractNodeClient" is incompatible with
    # definition in base class "LocationAwareObject"
    address: Address

    # QUESTION: These are incompatible with the LocationAwareObject properties
    network_id: Optional[Union[str, Any, Unspecified, All]]
    domain_id: Optional[Union[str, Any, Unspecified, All]]
    device_id: Optional[Union[str, Any, Unspecified, All]]
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

    def recv_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    def get_object(self) -> None:
        raise NotImplementedError

    def has_object(self) -> None:
        raise NotImplementedError

    def store_object(self) -> None:
        raise NotImplementedError

    def delete_object(self) -> None:
        raise NotImplementedError


class AbstractNodeClient:
    lib_ast: Any  # Cant import Globals (circular reference)
    address: Address
    """"""

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError
