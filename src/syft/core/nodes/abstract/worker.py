# -*- coding: utf-8 -*-
"""
stuff
"""

from __future__ import annotations

# NON-CORE IMPORTS
from ....common import AbstractWorker
from ....decorators import type_hints, syft_decorator
from typing import List
from ....util import get_subclasses

# CORE IMPORTS
from ...store.store import ObjectStore
from ...message import SyftMessage
from ...io import Route


class Worker(AbstractWorker):

    """
    Basic class for a syft worker behavior, explicit purpose workers will
    inherit this class (eg. WebsocketWorker, VirtualWorker).

    A worker is a collection of objects owned by a machine, a list of supported
    frameworks used for remote execution and a message router. The objects
    owned by the worker are placed in an ObjectStore object, the list of
    frameworks are a list of Globals and the message router is a dict that maps
    a message type to a processing method.

    Each worker is identified by an id of type str.
    """

    @syft_decorator(typechecking=True)
    def __init__(self, name: str = None):
        super().__init__()

        self.name = name
        self.store = ObjectStore()
        self.msg_router = {}

        self.services_registered = False

    @type_hints
    def recv_msg(self, msg: SyftMessage) -> SyftMessage:

        try:
            processed = self.msg_router[type(msg)].process(worker=self, msg=msg)
        except KeyError as e:
            if type(msg) not in self.msg_router:
                raise KeyError(
                    f"The worker {self.id} of type {type(self)} cannot process messages of type "
                    + f"{type(msg)} because there is no service running to process it."
                )
            else:

                if not self.services_registered:
                    raise Exception(
                        "Please call _register_services on worker. This seems to have"
                        "been skipped for some reason."
                    )

                raise e
        if self.shoud_forward(processed):
            self.forward_message(processed)
        return processed

    # TODO: change services type  to List[WorkerService] when typechecker allows subclassing
    @syft_decorator(typechecking=True)
    def _set_services(self, services: List) -> None:
        self.services = services
        self._register_services()

    @syft_decorator(typechecking=True)
    def _register_services(self) -> None:
        """In this method, we set each message type to the appropriate
        service for this worker. It's important to note that one message type
        cannot map to multiple services on any given worker type. If you want to
        send information to a different service, create a new message type for that
        service. Put another way, a service can have multiple message types which
        correspond to it, but each message type can only have one service (per worker
        subclass) which corresponds to it."""

        for s in self.services:
            service_instance = s()
            for handler_type in s.message_handler_types():
                self.msg_router[handler_type] = service_instance
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.msg_router[handler_type_subclass] = service_instance

        self.services_registered = True

    @type_hints
    def update_network(self) -> None:
        """
        This method allow connecting to a main orchestrator that can
        send the information about the surrounding networks to update
        the configuration jsons.
        """
        pass

    def sign_message(self, msg: SyftMessage) -> SyftMessage:
        """
        Add the worker's route to the message prior to forwarding
        to other entities on the network.
        """
        return msg

    @type_hints
    def listen_on_messages(self, msg: SyftMessage) -> SyftMessage:
        """
        Allows workers to connect to messaging protocols and listen on messages.
        """
        return self.recv_msg(msg)

    @type_hints
    def should_forward(self, msg:SyftMessage) -> boolean:
        """
        Determine if the message should be forwarded in the network
        """
        return True

    @type_hints
    def forward_message(self, msg: SyftMessage) -> None:
        """
        forward message to entities on the network that may have objects of interest.
        """
        signed_message = self.sign_message(msg)
        routes = []
        return self.broadcast_message(msg, routes)

    def broadcast_message(message: str, routes: Sequence[Route]) -> None:
        """
        This allows for broadcasting to either routes, or through a comms bus.
        """
        pass
