from typing import List

from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)
from syft.core.common.uid import UID
from ....decorators import syft_decorator
from ....lib import lib_ast
from ...io.address import Address
from ...io.route import Route
from ..abstract.node import AbstractNodeClient
from .location_aware_object import LocationAwareObject
from .service.child_node_lifecycle_service import RegisterChildNodeMessage


# TODO: Fix AbstractNode and LocationAwareObject being incompatible
class Client(AbstractNodeClient, LocationAwareObject):  # type: ignore # incompatible
    """Client is an incredibly powerful abstraction in Syft. We assume that,
    no matter where a client is, it can figure out how to communicate with
    the Node it is supposed to point to. If I send you a client I have
    with all of the metadata in it, you should have all the information
    you need to know to interact with a node (although you might not
    have permissions - clients should not store private keys)."""

    @syft_decorator(typechecking=True)
    def __init__(self, address: Address, name: str, routes: List[Route]):
        LocationAwareObject.__init__(self, address=address)

        self.name = name
        self.routes = routes
        self.default_route_index = 0

        self.install_supported_frameworks()

    def install_supported_frameworks(self) -> None:
        self.lib_ast = lib_ast.copy()
        self.lib_ast.set_client(self)

        for attr_name, attr in self.lib_ast.attrs.items():
            setattr(self, attr_name, attr)

    def add_me_to_my_address(self) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def register(self, client: AbstractNodeClient) -> None:
        msg = RegisterChildNodeMessage(child_node_client=client, address=self.address)
        self.send_immediate_msg_without_reply(msg=msg)

    @property
    def target_node_id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError

    @target_node_id.setter
    def target_node_id(self, new_target_node_id: UID) -> UID:
        """This client points to an node, this saves the id of that node"""
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply, route_index: int = 0
    ) -> ImmediateSyftMessageWithoutReply:
        route_index = route_index or self.default_route_index
        return self.routes[route_index].send_immediate_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply, route_index: int = 0
    ) -> None:
        route_index = route_index or self.default_route_index
        return self.routes[route_index].send_immediate_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply, route_index: int = 0
    ) -> None:
        route_index = route_index or self.default_route_index
        return self.routes[route_index].send_eventual_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        return f"<Client pointing to node with id:{self.target_node_id}>"

    @syft_decorator(typechecking=True)
    def register_route(self, route: Route) -> None:
        self.routes.append(route)

    @syft_decorator(typechecking=True)
    def set_default_route(self, route_index: int) -> None:
        self.default_route = route_index
