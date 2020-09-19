# -*- coding: utf-8 -*-
"""
stuff
"""

from __future__ import annotations

# NON-CORE IMPORTS
from ..common.node import AbstractNode
from ....decorators import syft_decorator
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
from ...io.address import Address
from ...io.address import address as create_address


class Node(AbstractNode):

    """
    Basic class for a syft node behavior, explicit purpose nodes will
    inherit this class (e.g., Device, Domain, Network, and VirtualMachine).



    Each node is identified by an id of type ID and a name of type string.
    """

    @syft_decorator(typechecking=True)
    def __init__(self, name: str = None, address: Address = None):
        super().__init__()

        # This is the name of the node - it exists purely to help the
        # end user have some idea about what this node is in a human
        # readable form. It is not guaranteed to be unique (or to
        # really be anything for that matter).
        self.name = name

        # All nodes should have a representation of where they think
        # they are currently held. Note that this is at risk of going
        # out of date and so we need to make sure we write good
        # logic to keep these addresses up to date. The main
        # way that it could go out of date is by the node being moved
        # by its parent or its parent being moved by a grandparent, etc.
        # without anyone telling this node. This would be bad because
        # it would mean that when the node creates a new Client for
        # someone to use, it might have trouble actually reaching
        # the node. Fortunately, the creation of a client is (always?)
        # going to be initiated by the parent node itself, so we should
        # be able to check for it there. TODO: did we check for it?
        if address is None:
            address = create_address()

        self.address = address

        # make sure address includes my own ID
        self.add_me_to_my_address()

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

    def add_me_to_my_address(self):
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: SyftMessage) -> bool:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def recv_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        if self.message_is_for_me(msg):
            print("the message is for me!!!")
            try:  # we use try/except here because it's marginally faster in Python
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

    @syft_decorator(typechecking=True)
    def recv_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:

        if self.message_is_for_me(msg):
            print("the message is for me!!!")
            try:  # we use try/except here because it's marginally faster in Python

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

    @syft_decorator(typechecking=True)
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
            # Create a single instance of the service to cache in the router corresponding
            # to one or more message types.
            service_instance = s()
            for handler_type in s.message_handler_types():

                # for each explicitly supported type, add it to the router
                self.msg_with_reply_router[handler_type] = service_instance

                # for all sub-classes of the explicitly supported type, add them
                # to the router as well.
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.msg_with_reply_router[handler_type_subclass] = service_instance

        for s in self.services_without_reply:
            # Create a single instance of the service to cache in the router corresponding
            # to one or more message types.
            service_instance = s()
            for handler_type in s.message_handler_types():

                # for each explicitly supported type, add it to the router
                self.msg_without_reply_router[handler_type] = service_instance

                # for all sub-classes of the explicitly supported type, add them
                # to the router as well.
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.msg_without_reply_router[
                        handler_type_subclass
                    ] = service_instance

        # Set the services_registered flag to true so that we know that all services
        # have been properly registered. This mostly exists because someone might
        # accidentally delete (forget to call) this method inside the __init__ function
        # of a sub-class of Node.
        self.services_registered = True
