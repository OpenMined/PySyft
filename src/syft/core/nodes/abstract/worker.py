# -*- coding: utf-8 -*-
"""
stuff
"""

from __future__ import annotations

# NON-CORE IMPORTS
from ....common import AbstractWorker
from ....ast.globals import Globals
from .... import type_hints

# CORE IMPORTS
from ...store.store import ObjectStore
from ...message import SyftMessage


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
        self.frameworks = Globals()
        self.msg_router = {}

        self._register_services()

    @type_hints
    def recv_msg(self, msg: SyftMessage) -> SyftMessage:
        try:
            return self.msg_router[type(msg)].process(worker=self, msg=msg)
        except KeyError as e:
            if(type(msg) not in self.msg_router):
                raise KeyError(f"The worker {self.id} of type {type(self)} cannot process messages of type "+
                f"{type(msg)} because there is no service running to process it.")
            else:
                raise e

    @type_hints
    def _register_services(self) -> None:
        raise NotImplementedError

