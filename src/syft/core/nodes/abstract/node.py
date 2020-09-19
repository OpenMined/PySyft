# -*- coding: utf-8 -*-
"""
stuff
"""

from __future__ import annotations

# NON-CORE IMPORTS
from ..common.node import AbstractNode
from ....decorators import type_hints, syft_decorator
from typing import List
from ....util import get_subclasses
from ...message.syft_message import SyftMessage
from ...message.syft_message import SyftMessageWithReply
from ...message.syft_message import SyftMessageWithoutReply

# CORE IMPORTS
from ...store.store import ObjectStore
from .service.msg_forwarding_service import MessageForwardingService
from .service.repr_service import ReprService
from .client import Client
from .service.heritage_update_service import HeritageUpdateService

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

        # This is the name of the node - it exists purely to help the
        # end user have some idea about what this node is in a human
        # readable form. It is not guaranteed to be unique (or to
        # really be anything for that matter).
        self.name = name

        # Any object that needs to be stored on a node is stored here
        # More specifically, all collections of objects are found here
        # There should be NO COLLECTIONS stored as attributes directly
        # on a Node if there is a chance that the collections could
        # become quite numerous (or otherwise fill up RAM).
        # self.store is the elastic memory.
        self.store = ObjectStore()

        # We need to register all the services once a node is created
        # On the off chance someone forgot to do this (super unlikely)
        # this flag exists to check it.
        self.services_registered = False

        # In order to be able to write generic services (in .service)
        # which can work for all node types, sometimes we need to have
        # a reference to what node type this node is. This attribute
        # provides that ability.
        self.node_type = type(self).__name__

        # ABOUT SERVICES AND MESSAGES

        # Each service corresponds to one or more message types which
        # the service processes. There are two kinds of messages, those
        # which require a reply and those which do not. Thus, there
        # are two kinds of services, service which generate a reply
        # and services which do not. It's important to distinguish
        # between them because:
        #
        # 1) services which do not return a reply
        # can typically be run on a more flexible time-table, whereas
        # services which require a reply often care about the latency
        # of the reply.
        #
        # 2) Services which do not return a reply aren't likely to leak
        # any information because no information is leaving. Thus, our
        # privacy/security concerns are more concentrated within service
        # which reply with some amount of information.


        # for messages which need a reply, this uses the type
        # of the message to look up the service which
        # addresses that message.
        self.msg_with_reply_router = {}

        # for messages which don't lead to a reply, this uses
        # the type of the message to look up the service
        # which addresses that message
        self.msg_without_reply_router = {}

        # for services which return a reply
        self.services_with_reply = list()

        # for services which do not return a reply
        self.services_without_reply = list()

        # This is the list of services which all nodes support.
        # You can read more about them by reading their respective
        # class documentation.
        self.services_without_reply.append(ReprService)
        self.services_without_reply.append(HeritageUpdateService)

        # This is a special service which cannot be listed in any
        # of the other services because it handles messages of all
        # types. Thus, it does not live in a message router since
        # routers only exist to decide which messages go to which
        # services, and they require that every message only correspond
        # to only one service type. If we have more messages like
        # these we'll make a special category for "services that
        # all messages are applied to" but for now this will do.
        self.message_forwarding_service = MessageForwardingService()

    @property
    def known_nodes(self) -> List[Client]:
        """This is a property which returns a list of all known nodes
        by returning the clients we used to interact with them from
        the object store."""

        return self.store.get_objects_of_type(obj_type=Client)

    @property
    def known_child_nodes(self):
        raise NotImplementedError

    def message_is_for_me(self, msg:SyftMessage) -> bool:
        raise NotImplementedError

    def recv_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        if self.message_is_for_me(msg):
            print("the message is for me!!!")
            try:
                return self.msg_with_reply_router[type(msg)].process(node=self, msg=msg)
            except KeyError as e:
                if type(msg) not in self.msg_with_reply_router:
                    raise KeyError(
                        f"The node {self.id} of type {type(self)} cannot process messages of type "
                        + f"{type(msg)} because there is no service running to process it."
                    )

                self.ensure_services_have_been_registered_error_if_not()
        else:
            print("the message is not for me...")
            return self.message_forwarding_service.process(node=self, msg=msg)

    def recv_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        if self.message_is_for_me(msg):
            print("the message is for me!!!")
            try:
                self.msg_without_reply_router[type(msg)].process(node=self, msg=msg)
            except KeyError as e:

                if type(msg) not in self.msg_without_reply_router:
                    raise KeyError(
                        f"The node {self.id} of type {type(self)} cannot process messages of type "
                        + f"{type(msg)} because there is no service running to process it."
                    )

                self.ensure_services_have_been_registered_error_if_not()

                raise e

        else:
            print("the message is not for me...")
            self.message_forwarding_service.process(node=self, msg=msg)

    def ensure_services_have_been_registered_error_if_not(self) -> None:
        if not self.services_registered:
            raise Exception(
                "Please call _register_services on node. This seems to have"
                "been skipped for some reason."
            )



    @syft_decorator(typechecking=True)
    def _register_services(self) -> None:
        """In this method, we set each message type to the appropriate
        service for this node. It's important to note that one message type
        cannot map to multiple services on any given node type. If you want to
        send information to a different service, create a new message type for that
        service. Put another way, a service can have multiple message types which
        correspond to it, but each message type can only have one service (per node
        subclass) which corresponds to it."""

        for s in self.services_with_reply:
            service_instance = s()
            for handler_type in s.message_handler_types():
                self.msg_with_reply_router[handler_type] = service_instance
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.msg_with_reply_router[handler_type_subclass] = service_instance

        for s in self.services_without_reply:
            service_instance = s()
            for handler_type in s.message_handler_types():
                self.msg_without_reply_router[handler_type] = service_instance
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.msg_without_reply_router[handler_type_subclass] = service_instance

        self.services_registered = True
