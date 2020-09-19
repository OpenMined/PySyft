# -*- coding: utf-8 -*-
"""
stuff
"""

from __future__ import annotations

# NON-CORE IMPORTS
from ....common import AbstractWorker
from ....ast.globals import Globals
from .... import type_hints
from typing import List

# CORE IMPORTS
from ...store.store import ObjectStore
from ...message import SyftMessage
from .service import WorkerService

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

    @type_hints
    def __init__(self, name: str = None):
        super().__init__()

        self.name = name
        self.store = ObjectStore()
        self.msg_router = {}

        self.services_registered = False


    @type_hints
    def recv_msg(self, msg: SyftMessage) -> SyftMessage:

        try:
            return self.msg_router[type(msg)].process(worker=self, msg=msg)
        except KeyError as e:
            if type(msg) not in self.msg_router:
                raise KeyError(
                    f"The worker {self.id} of type {type(self)} cannot process messages of type "
                    + f"{type(msg)} because there is no service running to process it."
                )
            else:

                if not self.services_registered:
                    raise Exception("Please call _register_services on worker. This seems to have"
                                    "been skipped for some reason.")

                raise e

    # TODO: change services type  to List[WorkerService] when typechecker allows subclassing
    @type_hints
    def _set_services(self, services: List) -> None:
        self.services = services

    @type_hints
    def _register_services(self) -> None:
        """In this method, we set each message type to the appropriate
        service for this worker. It's important to note that one message type
        cannot map to multiple services on any given worker type. If you want to
        send information to a different service, create a new message type for that
        service. Put another way, a service can have multiple message types which
        correspond to it, but each message type can only have one service (per worker
        subclass) which corresponds to it."""

        for s in self.services:
            for handler_type in s.message_handler_types():
                self.msg_router[handler_type] = s()
