# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import BaseSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base

# relative
from .... import __version__
from ....grid import GridURL
from ....lib import lib_ast
from ....logger import debug
from ....logger import error
from ....logger import info
from ....logger import traceback_and_raise
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
from ...io.location import Location
from ...io.location import SpecificLocation
from ...io.route import Route
from ...io.route import SoloRoute
from ...io.virtual import create_virtual_connection
from ..abstract.node import AbstractNode
from .action.exception_action import ExceptionMessage
from .action.exception_action import UnknownPrivateException
from .client import Client
from .metadata import Metadata
from .node_manager.redis_store import RedisStore
from .node_manager.setup_manager import SetupManager
from .node_service.auth import AuthorizationException
from .node_service.child_node_lifecycle.child_node_lifecycle_service import (
    ChildNodeLifecycleService,
)
from .node_service.get_repr.get_repr_service import GetReprService
from .node_service.heritage_update.heritage_update_service import HeritageUpdateService
from .node_service.msg_forwarding.msg_forwarding_service import (
    SignedMessageWithReplyForwardingService,
)
from .node_service.msg_forwarding.msg_forwarding_service import (
    SignedMessageWithoutReplyForwardingService,
)
from .node_service.node_credential.node_credential_messages import NodeCredentials
from .node_service.node_service import EventualNodeServiceWithoutReply
from .node_service.node_service import ImmediateNodeServiceWithReply
from .node_service.object_action.obj_action_service import (
    EventualObjectActionServiceWithoutReply,
)
from .node_service.object_action.obj_action_service import (
    ImmediateObjectActionServiceWithReply,
)
from .node_service.object_action.obj_action_service import (
    ImmediateObjectActionServiceWithoutReply,
)
from .node_service.object_search.obj_search_service import ImmediateObjectSearchService
from .node_service.object_search_permission_update.obj_search_permission_service import (
    ImmediateObjectSearchPermissionUpdateService,
)
from .node_service.resolve_pointer_type.resolve_pointer_type_service import (
    ResolvePointerTypeService,
)
from .node_service.testing_services.repr_service import ReprService
from .node_service.vpn.vpn_messages import VPNRegisterMessage
from .node_table import Base
from .node_table.node import Node as NodeRow

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

    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        TableBase: Any = None,
        db_engine: Any = None,
        store_type: type = RedisStore,
        settings: Optional[BaseSettings] = None,
    ):

        # The node has a name - it exists purely to help the
        # end user have some idea about what this node is in a human
        # readable form. It is not guaranteed to be unique (or to
        # really be anything for that matter).
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        self.settings = settings

        # TableBase is the base class from which all ORM classes must inherit
        # If one isn't provided then we can simply make one.
        if TableBase is None:
            TableBase = declarative_base()

        # If not provided a session connecting us to the database, let's just
        # initialize a database in memory
        if db_engine is None:
            db_engine = create_engine("sqlite://", echo=False)
            Base.metadata.create_all(db_engine)  # type: ignore

        # cache these variables on self
        self.TableBase = TableBase
        self.db_engine = db_engine
        # self.db = db
        # self.session = db

        # launch the tables in the database
        # Tudor: experimental
        # self.TableBase.metadata.create_all(engine)

        # Any object that needs to be stored on a node is stored here
        # More specifically, all collections of objects are found here
        # There should be NO COLLECTIONS stored as attributes directly
        # on a Node if there is a chance that the collections could
        # become quite numerous (or otherwise fill up RAM).
        # self.store is the elastic memory.

        self.store = store_type(db=self.db_engine, settings=settings)
        self.setup = SetupManager(database=self.db_engine)

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
        # which addresses that message.
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
        self.immediate_services_with_reply.append(GetReprService)
        self.immediate_services_with_reply.append(ResolvePointerTypeService)

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

        self.allowed_unsigned_messages = []
        self.allowed_unsigned_messages.append(VPNRegisterMessage)

        # now we need to load the relevant frameworks onto the node
        self.lib_ast = lib_ast
        # The node needs to sign messages that it sends so that recipients know that it
        # comes from the node. In order to do that, the node needs to generate keys
        # for itself to sign and verify with.

        # update keys
        if signing_key:
            Node.set_keys(node=self, signing_key=signing_key)

        # PERMISSION REGISTRY:
        self.guest_signing_key_registry = set()
        self.guest_verify_key_registry = set()
        self.admin_verify_key_registry = set()
        self.cpl_ofcr_verify_key_registry = set()
        self.peer_route_clients: Dict[UID, Dict[str, Dict[str, Client]]] = {}
        # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
        self.signaling_msgs = {}

        # For logging the number of messages received
        self.message_counter = 0

    def post_init(self) -> None:
        debug(f"> Creating {self.pprint}")

    def set_node_uid(self) -> None:
        try:
            setup = self.setup.first()
            # if its empty it will be set during CreateInitialSetUpMessage
            if setup.node_id != "":
                try:
                    node_id = UID.from_string(setup.node_id)
                except Exception as e:
                    error(f"Invalid Node UID in Setup Table. {setup.node_id}")
                    raise e

                location = SpecificLocation(name=setup.domain_name, id=node_id)
                # TODO: Fix with proper isinstance when the class will import
                if type(self).__name__ == "Domain":
                    self.domain = location
                elif type(self).__name__ == "Network":
                    self.network = location
                info(f"Finished setting Node UID. {location}")
            if setup.signing_key:
                signing_key = SigningKey(setup.signing_key, encoder=HexEncoder)
                Node.set_keys(node=self, signing_key=signing_key)
        except Exception:
            info("Setup hasnt run yet so ignoring set_node_uid")
            pass

    @staticmethod
    def set_keys(node: Node, signing_key: Optional[SigningKey] = None) -> None:
        if signing_key is None:
            signing_key = SigningKey.generate()

        node.signing_key = signing_key
        node.verify_key = signing_key.verify_key
        node.root_verify_key = node.verify_key  # TODO: CHANGE

    @property
    def icon(self) -> str:
        return "📍"

    def get_client(
        self,
        routes: Optional[List[Route]] = None,
        signing_key: Optional[SigningKey] = None,
    ) -> ClientT:
        if not routes:
            conn_client = create_virtual_connection(node=self)
            solo = SoloRoute(destination=self.target_id, connection=conn_client)
            # inject name
            setattr(
                solo,
                "name",
                f"Route ({self.name} <-> {self.name} Client)",
            )
            routes = [solo]

        return self.client_type(  # type: ignore
            name=self.name,
            routes=routes,
            network=self.network,
            domain=self.domain,
            device=self.device,
            vm=self.vm,
            signing_key=signing_key,  # If no signing_key is passed, the client generates one.
            verify_key=None,  # DO NOT PASS IN A VERIFY KEY!!! The client generates one.
        )

    def get_root_client(self, routes: Optional[List[Route]] = None) -> ClientT:
        client: ClientT = self.get_client(routes=routes)
        client.verify_key = self.verify_key
        client.signing_key = self.signing_key
        return client

    def get_metadata_for_client(self) -> Metadata:
        return Metadata(
            name=self.name if self.name else "",
            id=self.id,
            node=self.target_id,
            node_type=str(type(self).__name__),
            version=str(__version__),
        )

    def add_peer_routes(self, peer: NodeRow) -> None:
        try:
            routes = self.node_route.query(node_id=peer.id)  # type: ignore
            for route in routes:
                self.add_route(
                    node_id=UID.from_string(value=peer.node_uid),
                    node_name=peer.node_name,
                    host_or_ip=route.host_or_ip,
                    is_vpn=route.is_vpn,
                    private=route.private,
                    port=route.port,
                    protocol=route.protocol,
                )
        except Exception as e:
            error(f"Failed to add route to peer {peer}. {e}")

    def reload_peer_clients(self) -> None:
        peers = self.node.all()  # type: ignore
        for peer in peers:
            self.add_peer_routes(peer=peer)
        debug("Finished loading all the peer clients", self.peer_route_clients)

    def all_peer_clients(self) -> Dict[UID, List[Client]]:
        # get all the routes for each client and sort by VPN first
        all_clients = {}
        for node_id in self.peer_route_clients.keys():
            all_clients[node_id] = (
                list(self.peer_route_clients[node_id]["vpn"].values())
                + list(self.peer_route_clients[node_id]["https"].values())
                + list(self.peer_route_clients[node_id]["http"].values())
            )

        return all_clients

    def add_route(
        self,
        node_id: UID,
        node_name: str,
        host_or_ip: str,
        is_vpn: bool,
        private: bool,
        port: int,
        protocol: str,
    ) -> None:
        # relative
        from ....grid.client.client import connect

        debug(
            f"Adding route {node_id}, {node_name}, "
            + f"{protocol}://{host_or_ip}:{port}, vpn: {is_vpn}, private: {private}"
        )
        try:
            grid_url = GridURL.from_url(
                f"{protocol}://{host_or_ip}:{port}"
            ).as_container_host(container_host=self.settings.CONTAINER_HOST)
            security_key = "vpn" if is_vpn else protocol
            # make sure the node_id is in the Dict
            node_id_dict: Dict[str, Dict[str, Client]] = {
                "vpn": {},
                "http": {},
                "https": {},
            }
            if node_id in self.peer_route_clients:
                node_id_dict = self.peer_route_clients[node_id]

            if grid_url.base_url not in node_id_dict[security_key]:
                # connect and save the client
                client = connect(url=grid_url.with_path("/api/v1"), timeout=0.3)
                node_id_dict[security_key][grid_url.base_url] = client

            self.peer_route_clients[node_id] = node_id_dict
        except Exception as e:
            debug(
                f"Adding route {node_id}, {node_name}, "
                + f"{protocol}://{host_or_ip}:{port}, vpn: {is_vpn}, "
                + f"private: {private}. {e}"
            )

    def get_peer_client(self, node_id: UID, only_vpn: bool = True) -> Optional[Client]:
        # if we don't have it see if we can get it from the db first
        if node_id not in self.peer_route_clients:
            peer = self.node.first(node_uid=node_id.no_dash)  # type: ignore
            self.add_peer_routes(peer=peer)

        try:
            if node_id in self.peer_route_clients.keys():
                routes = self.peer_route_clients[node_id]
                # if we want VPN only then check there are some
                if only_vpn and "vpn" in routes and len(routes["vpn"]) == 0:
                    # we want VPN only but there are none
                    return None
                elif "vpn" in routes and len(routes["vpn"]) > 0:
                    # if we have VPN lets use it
                    return list(routes["vpn"].values())[0]
                elif "https" in routes and len(routes["https"]) > 0:
                    # we only have https
                    return list(routes["https"].values())[0]
                elif "http" in routes and len(routes["http"]) > 0:
                    # we only have http and don't care
                    return list(routes["http"].values())[0]
        except Exception as e:
            error(
                f"Exception while selecting node_id {node_id} from peer_route_clients. "
                f"{self.peer_route_clients}. {e}"
            )

        # there are no routes for this ID
        return None

    @property
    def id(self) -> UID:
        traceback_and_raise(NotImplementedError)

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        traceback_and_raise(NotImplementedError)

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        contents = getattr(msg, "message", msg)
        # exceptions can be easily triggered which break any loops
        # so we need to catch them here and respond with a special exception
        # message reply
        try:
            debug(
                f"> Received with Reply {contents.pprint} {contents.id} @ {self.pprint}"
            )
            # try to process message
            response = self.process_message(
                msg=msg, router=self.immediate_msg_with_reply_router
            )

        except Exception as e:
            print(type(e), e)
            error(e)
            public_exception: Exception
            if isinstance(e, AuthorizationException):
                private_log_msg = "An AuthorizationException has been triggered"
                public_exception = e
            else:
                private_log_msg = f"An {type(e)} has been triggered"  # dont send
                public_exception = UnknownPrivateException(str(e))
            try:
                # try printing a useful message
                private_log_msg += f" by {type(contents)} "
                private_log_msg += f"from {contents.reply_to}"  # type: ignore
            except Exception:
                error("Unable to format the private log message")
                pass
            # show the host what the real exception is
            error(private_log_msg)

            # send the public exception back
            response = ExceptionMessage(
                address=contents.reply_to,  # type: ignore
                msg_id_causing_exception=contents.id,
                exception_type=type(public_exception),
                exception_msg=str(public_exception),
            )

        # maybe I shouldn't have created process_message because it screws up
        # all the type inference.
        res_msg = response.sign(signing_key=self.signing_key)  # type: ignore
        output = (
            f"> {self.pprint} Signing {res_msg.pprint} with "
            + f"{self.key_emoji(key=self.signing_key.verify_key)}"  # type: ignore
        )
        debug(output)
        return res_msg

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        contents = getattr(msg, "message", msg)
        if contents:
            debug(
                f"> Received without Reply {contents.pprint} {contents.id} @ {self.pprint}"
            )

        self.process_message(msg=msg, router=self.immediate_msg_without_reply_router)
        try:
            pass
        except Exception as e:
            error(f"Exception processing {contents}. {e}")
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
                private_log_msg += f" by {type(contents)} "
                private_log_msg += f"from {contents.reply_to}"  # type: ignore
            except Exception:
                error("Unable to format the private log message")
                pass
            # show the host what the real exception is
            error(private_log_msg)

            # we still want to raise for now due to certain exceptions we expect
            # in tests
            if not isinstance(e, DuplicateRequestException):
                error(e)
                # TODO: A lot of tests are depending on this raise which seems bad
                traceback_and_raise(e)

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

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        self.process_message(msg=msg, router=self.eventual_msg_without_reply_router)

    # TODO: Add SignedEventualSyftMessageWithoutReply and others
    def process_message(
        self, msg: SignedMessage, router: dict
    ) -> Union[SyftMessage, None]:
        self.message_counter += 1
        try:
            contents = getattr(
                msg, "message", msg
            )  # in the event the message is unsigned
            debug(f"> Processing 📨 {msg.pprint} @ {self.pprint} {contents}")
            if self.message_is_for_me(msg=msg):
                debug(
                    f"> Recipient Found {msg.pprint}{msg.address.target_emoji()} == {self.pprint}"
                )

                # only a small number of messages are allowed to be unsigned otherwise
                # they need to be valid
                if type(msg) not in self.allowed_unsigned_messages and not msg.is_valid:  # type: ignore
                    error(f"Message is not valid. {msg}")
                    traceback_and_raise(Exception("Message is not valid."))

                # Process Message here
                try:  # we use try/except here because it's marginally faster in Python
                    service = router[type(contents)]
                except KeyError as e:
                    log = (
                        f"The node {self.id} of type {type(self)} cannot process messages of type "
                        + f"{type(contents)} because there is no service running to process it."
                        + f"{e}"
                    )
                    error(log)
                    self.ensure_services_have_been_registered_error_if_not()
                    traceback_and_raise(KeyError(log))

                if type(msg) in self.allowed_unsigned_messages:  # type: ignore
                    result = service.process(node=self, msg=contents, verify_key=None)
                else:
                    result = service.process(
                        node=self,
                        msg=contents,
                        verify_key=msg.verify_key,
                    )
                return result

            else:
                debug(
                    f"> Recipient Not Found ↪️ {msg.pprint}{msg.address.target_emoji()} != {self.pprint}"
                )
                # Forward message onwards
                if issubclass(type(msg), SignedImmediateSyftMessageWithReply):
                    return self.signed_message_with_reply_forwarding_service.process(
                        node=self,
                        msg=msg,  # type: ignore
                    )
                if issubclass(type(msg), SignedImmediateSyftMessageWithoutReply):
                    return self.signed_message_without_reply_forwarding_service.process(
                        node=self,
                        msg=msg,  # type: ignore
                    )
        except Exception as e:
            error(e)
            raise e
        return None

    def ensure_services_have_been_registered_error_if_not(self) -> None:
        if not self.services_registered:
            traceback_and_raise(
                Exception(
                    "Please call _register_services on node. This seems to have"
                    "been skipped for some reason."
                )
            )

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
        no_dash = str(self.id).replace("-", "")
        return f"{self.node_type}: {self.name}: {no_dash}"

    def get_credentials(self) -> NodeCredentials:
        return NodeCredentials.from_objs(
            node_uid=self.id,
            node_name=self.name,
            node_type=self.node_type,
            verify_key=self.verify_key,
        )
