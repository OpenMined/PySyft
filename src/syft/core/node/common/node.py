# -*- coding: utf-8 -*-
"""
CODING GUIDELINES:

Do NOT (without talking to trask):
- add another high level method for sending or receiving messages (like recv_eventual_msg_without_reply)
- add a service to the list of services below unless you're SURE all nodes will need it!
- serialize anything with pickle
"""

# stdlib
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# syft relative
from ....decorators import syft_decorator
from ....lib import lib_ast
from ....util import get_subclasses
from ...common.message import EventualSyftMessageWithoutReply
from ...common.message import ImmediateSyftMessageWithReply
from ...common.message import ImmediateSyftMessageWithoutReply
from ...common.message import SignedEventualSyftMessageWithoutReply
from ...common.message import SignedImmediateSyftMessageWithReply
from ...common.message import SignedImmediateSyftMessageWithoutReply
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.address import Address
from ...io.location import Location
from ...io.route import Route
from ...io.route import SoloRoute
from ...io.virtual import create_virtual_connection
from ...store import MemoryStore
from ..abstract.node import AbstractNode
from .action.exception_action import ExceptionMessage
from .action.exception_action import UnknownPrivateException
from .client import Client
from .service.auth import AuthorizationException
from .service.child_node_lifecycle_service import ChildNodeLifecycleService
from .service.heritage_update_service import HeritageUpdateService
from .service.msg_forwarding_service import SignedMessageWithReplyForwardingService
from .service.msg_forwarding_service import SignedMessageWithoutReplyForwardingService
from .service.node_service import EventualNodeServiceWithoutReply
from .service.node_service import ImmediateNodeServiceWithReply
from .service.obj_action_service import EventualObjectActionServiceWithoutReply
from .service.obj_action_service import ImmediateObjectActionServiceWithReply
from .service.obj_action_service import ImmediateObjectActionServiceWithoutReply
from .service.obj_search_permission_service import (
    ImmediateObjectSearchPermissionUpdateService,
)
from .service.obj_search_service import ImmediateObjectSearchService
from .service.repr_service import ReprService

# this generic type for Client bound by Client
ClientT = TypeVar("ClientT", bound=Client)


# TODO: Move but right now import loop prevents importing from the RequestMessage
class DuplicateRequestException(Exception):
    pass


class Node(AbstractNode):

    """
    Basic class for a syft node behavior, explicit purpose node will
    inherit this class (e.g., Device, Domain, Network, and VirtualMachine).

    Each node is identified by an id of type ID and a name of type string.
    """

    client_type = ClientT
    child_type_client_type = ClientT

    ChildT = TypeVar("ChildT", bound="Node")
    child_type = ChildT

    signing_key: Optional[SigningKey]
    verify_key: Optional[VerifyKey]

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
    ):

        # The node has a name - it exists purely to help the
        # end user have some idea about what this node is in a human
        # readable form. It is not guaranteed to be unique (or to
        # really be anything for that matter).
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

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
        self.immediate_msg_with_reply_router: Dict[
            Type[ImmediateSyftMessageWithReply], ImmediateNodeServiceWithReply
        ] = {}

        # for messages which don't lead to a reply, this uses
        # the type of the message to look up the service
        # which addresses that message
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

        # TODO: Support ImmediateNodeServiceWithoutReply Parent Class
        # for services which run immediately and do not return a reply
        self.immediate_services_without_reply: List[Any] = []
        self.immediate_services_without_reply.append(ReprService)
        self.immediate_services_without_reply.append(HeritageUpdateService)
        self.immediate_services_without_reply.append(ChildNodeLifecycleService)
        self.immediate_services_without_reply.append(
            ImmediateObjectActionServiceWithoutReply
        )
        self.immediate_services_without_reply.append(
            ImmediateObjectSearchPermissionUpdateService
        )

        # TODO: Support ImmediateNodeServiceWithReply Parent Class
        # for services which run immediately and return a reply
        self.immediate_services_with_reply: List[Any] = []
        self.immediate_services_with_reply.append(ImmediateObjectActionServiceWithReply)
        self.immediate_services_with_reply.append(ImmediateObjectSearchService)

        # for services which can run at a later time and do not return a reply
        self.eventual_services_without_reply = list()
        self.eventual_services_without_reply.append(
            EventualObjectActionServiceWithoutReply
        )

        # This is a special service which cannot be listed in any
        # of the other services because it handles messages of all types.
        # Thus, it does not live in a message router since
        # routers only exist to decide which messages go to which
        # services, and they require that every message only correspond
        # to only one service type. If we have more messages like
        # these we'll make a special category for "services that
        # all messages are applied to" but for now this will do.

        self.signed_message_with_reply_forwarding_service = (
            SignedMessageWithReplyForwardingService()
        )

        self.signed_message_without_reply_forwarding_service = (
            SignedMessageWithoutReplyForwardingService()
        )

        # now we need to load the relevant frameworks onto the node
        self.lib_ast = lib_ast

        # The node needs to sign messages that it sends so that recipients know that it
        # comes from the node. In order to do that, the node needs to generate keys
        # for itself to sign and verify with.

        # create a signing key if one isn't provided
        if signing_key is None:
            self.signing_key = SigningKey.generate()
        else:
            self.signing_key = signing_key

        # if verify key isn't provided, get verify key from signing key
        if verify_key is None:
            self.verify_key = self.signing_key.verify_key
        else:
            self.verify_key = verify_key

        # PERMISSION REGISTRY:
        self.root_verify_key = self.verify_key  # TODO: CHANGE
        self.guest_verify_key_registry = set()
        self.in_memory_client_registry = {}
        # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
        self.signaling_msgs = {}

        # For logging the number of messages received
        self.message_counter = 0

    @property
    def icon(self) -> str:
        return "📍"

    @syft_decorator(typechecking=True)
    def get_client(self, routes: List[Route] = []) -> ClientT:
        if not len(routes):
            conn_client = create_virtual_connection(node=self)
            solo = SoloRoute(destination=self.target_id, connection=conn_client)
            # inject name
            setattr(solo, "name", f"Route ({self.name} <-> {self.name} Client)")
            routes = [solo]

        return self.client_type(
            name=self.name,
            routes=routes,
            network=self.network,
            domain=self.domain,
            device=self.device,
            vm=self.vm,
            signing_key=None,  # DO NOT PASS IN A SIGNING KEY!!! The client generates one.
            verify_key=None,  # DO NOT PASS IN A VERIFY KEY!!! The client generates one.
        )

    @syft_decorator(typechecking=True)
    def get_root_client(self, routes: List[Route] = []) -> ClientT:
        client = self.get_client(routes=routes)
        self.root_verify_key = client.verify_key
        return client

    def get_metadata_for_client(self) -> str:
        metadata: Dict[str, Union[Address, Optional[str], Location]] = {}

        metadata["spec_location"] = self.target_id.json()
        metadata["name"] = self.name
        metadata["id"] = self.id.json()

        return json.dumps(metadata)

    @property
    def known_nodes(self) -> List[Client]:
        """This is a property which returns a list of all known node
        by returning the clients we used to interact with them from
        the object store."""
        return list(self.in_memory_client_registry.values())

    @property
    def id(self) -> UID:
        raise NotImplementedError

    @property
    def known_child_nodes(self) -> List[Address]:
        if sy.VERBOSE:
            print(f"> {self.pprint} Getting known Children Nodes")
        if self.child_type_client_type is not None:
            return [
                client
                for client in self.in_memory_client_registry.values()
                if all(
                    [
                        self.network is None
                        or client.network is None
                        or self.network == client.network,
                        self.domain is None
                        or client.domain is None
                        or self.domain == client.domain,
                        self.device is None
                        or client.device is None
                        or self.device == client.device,
                        self.vm is None or client.vm is None or self.vm == client.vm,
                    ]
                )
            ]
        else:
            if sy.VERBOSE:
                print(f"> Node {self.pprint} has no children")
            return []

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        # exceptions can be easily triggered which break any WebRTC loops
        # so we need to catch them here and respond with a special exception
        # message reply
        try:
            # try to process message
            response = self.process_message(
                msg=msg, router=self.immediate_msg_with_reply_router
            )

        except Exception as e:
            public_exception: Exception
            if isinstance(e, AuthorizationException):
                private_log_msg = "An AuthorizationException has been triggered"
                public_exception = e
            else:
                private_log_msg = f"An {type(e)} has been triggered"  # dont send
                public_exception = UnknownPrivateException(
                    "UnknownPrivateException has been triggered."
                )
            try:
                # try printing a useful message
                private_log_msg += f" by {type(msg.message)} "
                private_log_msg += f"from {msg.message.reply_to}"  # type: ignore
            except Exception:
                pass
            # show the host what the real exception is
            print(private_log_msg)

            # send the public exception back
            response = ExceptionMessage(
                address=msg.message.reply_to,  # type: ignore
                msg_id_causing_exception=msg.message.id,
                exception_type=type(public_exception),
                exception_msg=str(public_exception),
            )

        # maybe I shouldn't have created process_message because it screws up
        # all the type inference.
        res_msg = response.sign(signing_key=self.signing_key)  # type: ignore
        if sy.VERBOSE:
            output = (
                f"> {self.pprint} Signing {res_msg.pprint} with "
                + f"{self.key_emoji(key=self.signing_key.verify_key)}"  # type: ignore
            )
            print(output)
        return res_msg

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        if sy.VERBOSE:
            print(f"> Received {msg.pprint} @ {self.pprint}")
        try:
            self.process_message(
                msg=msg, router=self.immediate_msg_without_reply_router
            )
        except Exception as e:
            # public_exception: Exception
            if isinstance(e, DuplicateRequestException):
                private_log_msg = "An DuplicateRequestException has been triggered"
                # public_exception = e
            else:
                private_log_msg = f"An {type(e)} has been triggered"  # dont send
                # public_exception = UnknownPrivateException(
                #     "UnknownPrivateException has been triggered."
                # )
            try:
                # try printing a useful message
                private_log_msg += f" by {type(msg.message)} "
                private_log_msg += f"from {msg.message.reply_to}"  # type: ignore
            except Exception:
                pass
            # show the host what the real exception is
            print(private_log_msg)

            # we still want to raise for now due to certain exceptions we expect
            # in tests
            if not isinstance(e, DuplicateRequestException):
                raise e

            # TODO: finish code to send ExceptionMessage back
            # if isinstance(e, DuplicateRequestException):
            #     # we have a reply_to
            #     # send the public exception back
            #     response = ExceptionMessage(
            #         address=msg.message.reply_to,  # type: ignore
            #         msg_id_causing_exception=msg.message.id,
            #         exception_type=type(public_exception),
            #         exception_msg=str(public_exception),
            #     )
            #     res_msg = response.sign(signing_key=self.signing_key)  # type: ignore
            #     self.client.send_immediate_msg_with_reply.process(
            #         node=self,
            #         msg=res_msg,
            #     )
        return None

    @syft_decorator(typechecking=True)
    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        self.process_message(msg=msg, router=self.eventual_msg_without_reply_router)

    # TODO: Add SignedEventualSyftMessageWithoutReply and others
    def process_message(
        self, msg: SignedMessage, router: dict
    ) -> Union[SyftMessage, None]:

        self.message_counter += 1

        if sy.VERBOSE:
            print(f"> Processing 📨 {msg.pprint} @ {self.pprint}")
        if self.message_is_for_me(msg=msg):
            if sy.VERBOSE:
                print(
                    f"> Recipient Found {msg.pprint}{msg.address.target_emoji()} == {self.pprint}"
                )
            # Process Message here
            if not msg.is_valid:
                raise Exception("Message is not valid.")

            try:  # we use try/except here because it's marginally faster in Python
                service = router[type(msg.message)]

            except KeyError as e:
                self.ensure_services_have_been_registered_error_if_not()

                raise KeyError(
                    f"The node {self.id} of type {type(self)} cannot process messages of type "
                    + f"{type(msg.message)} because there is no service running to process it."
                    + f"{e}"
                )

            return service.process(
                node=self,
                msg=msg.message,
                verify_key=msg.verify_key,
            )

        else:
            if sy.VERBOSE:
                print(
                    f"> Recipient Not Found ↪️ {msg.pprint}{msg.address.target_emoji()} != {self.pprint}"
                )
            # Forward message onwards
            if issubclass(type(msg), SignedImmediateSyftMessageWithReply):
                return self.signed_message_with_reply_forwarding_service.process(
                    node=self,
                    msg=msg,
                )
            if issubclass(type(msg), SignedImmediateSyftMessageWithoutReply):
                return self.signed_message_without_reply_forwarding_service.process(
                    node=self,
                    msg=msg,
                )
        return None

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

        for isr in self.immediate_services_with_reply:
            # Create a single instance of the service to cache in the router corresponding
            # to one or more message types.
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
            # to one or more message types.
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
            # to one or more message types.
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
