# -*- coding: utf-8 -*-
"""
stuff
"""

from __future__ import annotations
import json

# NON-CORE IMPORTS
from ....common import AbstractNode
from ....decorators import type_hints, syft_decorator
from typing import List
from ....util import get_subclasses

# CORE IMPORTS
from ...store.store import ObjectStore
from ...message import SyftMessage
from ...io import Route

# nodes related imports
from ..abstract.remote_nodes import RemoteNodes


class Node(AbstractNode):

    """
    Basic class for a syft node behavior, explicit purpose nodes will
    inherit this class (e.g., Device, Domain, Network, and VirtualMachine).

    A node is a collection of objects owned by a machine, a list of supported
    frameworks used for remote execution and a message router. The objects
    owned by the node are placed in an ObjectStore object, the list of
    frameworks are a list of Globals and the message router is a dict that maps
    a message type to a processing method.

    Each node is identified by an id of type ID and a name of type string.
    """

    @syft_decorator(typechecking=True)
    def __init__(self, name: str = None):
        super().__init__()

        self.name = name
        self.store = ObjectStore()
        self.msg_router = {}
        # bootstrap
        self.known_workers = RemoteNodes()
        self.services_registered = False

    @type_hints
    def recv_msg(self, msg: SyftMessage) -> SyftMessage:
        try:
            return self.msg_router[type(msg)].process(worker=self, msg=msg)
        except KeyError as e:
            if type(msg) not in self.msg_router:
                raise KeyError(
                    f"The node {self.id} of type {type(self)} cannot process messages of type "
                    + f"{type(msg)} because there is no service running to process it."
                )
            else:

                if not self.services_registered:
                    raise Exception(
                        "Please call _register_services on node. This seems to have"
                        "been skipped for some reason."
                    )

                raise e

    # TODO: change services type  to List[NodeService] when typechecker allows subclassing
    @syft_decorator(typechecking=True)
    def _set_services(self, services: List) -> None:
        self.services = services
        self._register_services()

    @syft_decorator(typechecking=True)
    def _register_services(self) -> None:
        """In this method, we set each message type to the appropriate
        service for this node. It's important to note that one message type
        cannot map to multiple services on any given node type. If you want to
        send information to a different service, create a new message type for that
        service. Put another way, a service can have multiple message types which
        correspond to it, but each message type can only have one service (per node
        subclass) which corresponds to it."""

        for s in self.services:
            service_instance = s()
            for handler_type in s.message_handler_types():
                self.msg_router[handler_type] = service_instance
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.msg_router[handler_type_subclass] = service_instance

        self.services_registered = True

    @type_hints
    def listen_on_messages(self, msg: SyftMessage) -> SyftMessage:
        """
        Allows workers to connect to open messaging protocols and listen on
            messages.

        The worker would extend this class to implement the specific protocol.
        """
        return self.recv_msg(msg)
