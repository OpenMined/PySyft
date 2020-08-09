# -*- coding: utf-8 -*-
"""
stuff
"""

from typing import List, TypeVar, Dict, Union, Optional, Type, Any

from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
    SyftMessage,
)

from ....decorators import syft_decorator
from ....lib import lib_ast
from ....util import get_subclasses
from ...io.address import Address
from ...io.location import Location
from ...io.virtual import create_virtual_connection
from ...io.route import SoloRoute, Route

# CORE IMPORTS
from ...store import MemoryStore
from ...common.object import UID

# NON-CORE IMPORTS
from ..abstract.node import AbstractNode
from .client import Client
from .service.child_node_lifecycle_service import ChildNodeLifecycleService
from .service.heritage_update_service import HeritageUpdateService
from .service.msg_forwarding_service import (
    MessageWithoutReplyForwardingService,
    MessageWithReplyForwardingService,
)
from .service.obj_action_service import (
    EventualObjectActionServiceWithoutReply,
    ImmediateObjectActionServiceWithoutReply,
    ImmediateObjectActionServiceWithReply,
)
from .service.repr_service import ReprService
from .service.node_service import EventualNodeServiceWithoutReply


# TODO: Fix AbstractNode and LocationAwareObject being incompatible
class Node(AbstractNode):

    """
    Basic class for a syft node behavior, explicit purpose node will
    inherit this class (e.g., Device, Domain, Network, and VirtualMachine).

    Each node is identified by an id of type ID and a name of type string.
    """

    ClientT = TypeVar("ClientT", bound=Client)
    client_type = ClientT
    child_type_client_type = ClientT

    ChildT = TypeVar("ChildT", bound="Node")
    child_type = ChildT

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):

        super().__init__(network=network, domain=domain, device=device, vm=vm)

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
        self.store = MemoryStore()

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

        # Each service corresponds to one or more old_message types which
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
        # of the old_message to look up the service which
        # addresses that old_message.
        self.immediate_msg_with_reply_router: Dict[
            Type[ImmediateSyftMessageWithReply], ImmediateObjectActionServiceWithReply
        ] = {}

        # for messages which don't lead to a reply, this uses
        # the type of the old_message to look up the service
        # which addresses that old_message
        self.immediate_msg_without_reply_router: Dict[
            Type[ImmediateSyftMessageWithoutReply], Any
        ] = {}

        # for messages which don't need to be run right now
        # and will not generate a reply.
        self.eventual_msg_without_reply_router: Dict[
            Type[EventualSyftMessageWithoutReply], EventualNodeServiceWithoutReply
        ] = {}

        # This is the list of services which all node support.
        # You can read more about them by reading their respective
        # class documentation.

        # for services which run immediately and do not return a reply

        # TODO: Support ImmediateNodeServiceWithoutReply Parent Class
        self.immediate_services_without_reply: List[Any] = []
        self.immediate_services_without_reply.append(ReprService)
        self.immediate_services_without_reply.append(HeritageUpdateService)
        self.immediate_services_without_reply.append(ChildNodeLifecycleService)
        self.immediate_services_without_reply.append(
            ImmediateObjectActionServiceWithoutReply
        )

        # for services which run immediately and return a reply
        self.immediate_services_with_reply = list()
        self.immediate_services_with_reply.append(ImmediateObjectActionServiceWithReply)

        # for services which can run at a later time and do not return a reply
        self.eventual_services_without_reply = list()
        self.eventual_services_without_reply.append(
            EventualObjectActionServiceWithoutReply
        )

        # This is a special service which cannot be listed in any
        # of the other services because it handles messages of all types.
        # Thus, it does not live in a old_message router since
        # routers only exist to decide which messages go to which
        # services, and they require that every old_message only correspond
        # to only one service type. If we have more messages like
        # these we'll make a special category for "services that
        # all messages are applied to" but for now this will do.
        self.message_without_reply_forwarding_service = (
            MessageWithoutReplyForwardingService()
        )
        self.message_with_reply_forwarding_service = MessageWithReplyForwardingService()

        # now we need to load the relevant frameworks onto the node
        self.lib_ast = lib_ast

    @syft_decorator(typechecking=True)
    def get_client(self, routes: List[Route] = []) -> Client:
        if not len(routes):
            conn_client = create_virtual_connection(node=self)
            routes = [SoloRoute(destination=self.id, connection=conn_client)]
        return self.client_type(
            name=self.name,
            routes=routes,
            network=self.network,
            domain=self.domain,
            device=self.device,
            vm=self.vm,
        )

    def get_metadata_for_client(self) -> Dict[str, Union[Address, Optional[str], UID]]:
        metadata: Dict[str, Union[Address, Optional[str], UID]] = {}
        metadata["address"] = self
        metadata["name"] = self.name
        metadata["id"] = self.id
        return metadata

    @property
    def known_nodes(self) -> List[Client]:
        """This is a property which returns a list of all known node
        by returning the clients we used to interact with them from
        the object store."""

        # QUESTION: where is get_objects_of_type defined?
        return self.store.get_objects_of_type(obj_type=Client)

    @property
    def id(self):
        NotImplementedError

    @property
    def known_child_nodes(self) -> List:
        if self.child_type_client_type is not None:
            return self.store.get_objects_of_type(obj_type=self.child_type_client_type)
        else:
            return []

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: SyftMessage) -> bool:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        if self.message_is_for_me(msg=msg):
            print("the old_message is for me!!!")
            try:  # we use try/except here because it's marginally faster in Python
                return self.immediate_msg_with_reply_router[type(msg)].process(
                    node=self, msg=msg
                )
            except KeyError as e:
                if type(msg) not in self.immediate_msg_with_reply_router:
                    raise KeyError(
                        f"The node {self.id} of type {type(self)} cannot process messages of type "
                        + f"{type(msg)} because there is no service running to process it."
                        + f"{e}"
                    )

                self.ensure_services_have_been_registered_error_if_not()
        else:
            print("the old_message is not for me...")
            return self.message_with_reply_forwarding_service.process(
                node=self, msg=msg
            )

        # QUESTION what is the preferred pattern here, as the above doesnt exhaustively
        # return
        raise Exception("Unable to dispatch message, not all exceptions were caught")

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:

        if self.message_is_for_me(msg=msg):
            print("the message is for me!!!")
            try:  # we use try/except here because it's marginally faster in Python

                self.immediate_msg_without_reply_router[type(msg)].process(
                    node=self, msg=msg
                )

            except KeyError as e:

                if type(msg) not in self.immediate_msg_without_reply_router:
                    raise KeyError(
                        f"The node {self.id} of type {type(self)} cannot process messages of type "
                        + f"{type(msg)} because there is no service running to process it."
                    )

                self.ensure_services_have_been_registered_error_if_not()

                raise e

        else:
            print("the message is not for me...")
            self.message_without_reply_forwarding_service.process(node=self, msg=msg)

    @syft_decorator(typechecking=True)
    def recv_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:

        if self.message_is_for_me(msg=msg):
            print("the old_message is for me!!!")
            try:  # we use try/except here because it's marginally faster in Python

                self.eventual_msg_without_reply_router[type(msg)].process(
                    node=self, msg=msg
                )

            except KeyError as e:

                if type(msg) not in self.eventual_msg_without_reply_router:
                    raise KeyError(
                        f"The node {self.id} of type {type(self)} cannot process messages of type "
                        + f"{type(msg)} because there is no service running to process it."
                    )

                self.ensure_services_have_been_registered_error_if_not()

                raise e

        else:
            print("the old_message is not for me...")
            self.message_without_reply_forwarding_service.process(node=self, msg=msg)

    @syft_decorator(typechecking=True)
    def ensure_services_have_been_registered_error_if_not(self) -> None:
        if not self.services_registered:
            raise Exception(
                "Please call _register_services on node. This seems to have"
                "been skipped for some reason."
            )

    @syft_decorator(typechecking=True)
    def _register_services(self) -> None:
        """In this method, we set each old_message type to the appropriate
        service for this node. It's important to note that one old_message type
        cannot map to multiple services on any given node type. If you want to
        send information to a different service, create a new old_message type for that
        service. Put another way, a service can have multiple old_message types which
        correspond to it, but each old_message type can only have one service (per node
        subclass) which corresponds to it."""

        for isr in self.immediate_services_with_reply:
            # Create a single instance of the service to cache in the router corresponding
            # to one or more old_message types.
            isr_instance = isr()
            for handler_type in isr.message_handler_types():

                # for each explicitly supported type, add it to the router
                self.immediate_msg_with_reply_router[handler_type] = isr_instance

                # for all sub-classes of the explicitly supported type, add them
                # to the router as well.
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.immediate_msg_with_reply_router[
                        handler_type_subclass
                    ] = isr_instance

        for iswr in self.immediate_services_without_reply:
            # Create a single instance of the service to cache in the router corresponding
            # to one or more old_message types.
            iswr_instance = iswr()
            for handler_type in iswr.message_handler_types():

                # for each explicitly supported type, add it to the router
                self.immediate_msg_without_reply_router[handler_type] = iswr_instance

                # for all sub-classes of the explicitly supported type, add them
                # to the router as well.
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.immediate_msg_without_reply_router[
                        handler_type_subclass
                    ] = iswr_instance

        for eswr in self.eventual_services_without_reply:
            # Create a single instance of the service to cache in the router corresponding
            # to one or more old_message types.
            eswr_instance = eswr()
            for handler_type in eswr.message_handler_types():

                # for each explicitly supported type, add it to the router
                self.eventual_msg_without_reply_router[handler_type] = eswr_instance

                # for all sub-classes of the explicitly supported type, add them
                # to the router as well.
                for handler_type_subclass in get_subclasses(obj_type=handler_type):
                    self.eventual_msg_without_reply_router[
                        handler_type_subclass
                    ] = eswr_instance

        # Set the services_registered flag to true so that we know that all services
        # have been properly registered. This mostly exists because someone might
        # accidentally delete (forget to call) this method inside the __init__ function
        # of a sub-class of Node.
        self.services_registered = True

    def __repr__(self) -> str:
        return f"{self.node_type}:{self.name}:{self.id}"
