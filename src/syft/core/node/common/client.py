# stdlib
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pandas as pd

# syft relative
from .... import serialize
from ....lib import create_lib_ast
from ....logger import critical
from ....logger import debug
from ....logger import error
from ....logger import traceback_and_raise
from ....proto.core.node.common.client_pb2 import Client as Client_PB
from ....proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB
from ....util import get_fully_qualified_name
from ...common.message import EventualSyftMessageWithoutReply
from ...common.message import ImmediateSyftMessageWithReply
from ...common.message import ImmediateSyftMessageWithoutReply
from ...common.message import SignedEventualSyftMessageWithoutReply
from ...common.message import SignedImmediateSyftMessageWithReply
from ...common.message import SignedImmediateSyftMessageWithoutReply
from ...common.message import SyftMessage
from ...common.serde.deserialize import _deserialize
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ...io.route import Route
from ...io.route import SoloRoute
from ...io.virtual import VirtualClientConnection
from ...node.common.service.obj_search_service import ObjectSearchMessage
from ...pointer.garbage_collection import GarbageCollection
from ...pointer.garbage_collection import gc_get_default_strategy
from ...pointer.pointer import Pointer
from ..abstract.node import AbstractNodeClient
from .action.exception_action import ExceptionMessage
from .service.child_node_lifecycle_service import RegisterChildNodeMessage


class Client(AbstractNodeClient):
    """Client is an incredibly powerful abstraction in Syft. We assume that,
    no matter where a client is, it can figure out how to communicate with
    the Node it is supposed to point to. If I send you a client I have
    with all of the metadata in it, you should have all the information
    you need to know to interact with a node (although you might not
    have permissions - clients should not store private keys)."""

    def __init__(
        self,
        name: Optional[str],
        routes: List[Route],
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
    ):
        name = f"{name} Client" if name is not None else None
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        self.routes = routes
        self.default_route_index = 0

        gc_strategy_name = gc_get_default_strategy()
        self.gc = GarbageCollection(gc_strategy_name)

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

        self.install_supported_frameworks()

        self.store = StoreClient(client=self)

    @property
    def icon(self) -> str:
        icon = "📡"
        sub = []
        if self.vm is not None:
            sub.append("🍰")
        if self.device is not None:
            sub.append("📱")
        if self.domain is not None:
            sub.append("🏰")
        if self.network is not None:
            sub.append("🔗")

        if len(sub) > 0:
            icon = f"{icon} ["
            for s in sub:
                icon += s
            icon += "]"
        return icon

    @staticmethod
    def deserialize_client_metadata_from_node(
        metadata: Metadata_PB,
    ) -> Tuple[SpecificLocation, str, UID]:
        # string of bytes
        meta = _deserialize(blob=metadata)
        return meta.node, meta.name, meta.id

    def install_supported_frameworks(self) -> None:
        self.lib_ast = create_lib_ast(client=self)

        # first time we want to register for future updates
        self.lib_ast.register_updates(self)

        if self.lib_ast is not None:
            for attr_name, attr in self.lib_ast.attrs.items():
                setattr(self, attr_name, attr)

        # shortcut syft.lib.python to just python
        if hasattr(self.lib_ast, "syft"):
            try:
                lib_attr = getattr(self.lib_ast.syft, "lib", None)

                if lib_attr is not None:
                    python_attr = getattr(lib_attr, "python", None)
                    setattr(self, "python", python_attr)

            except Exception as e:
                critical(f"Failed to set python attribute on client. {e}")

    def add_me_to_my_address(self) -> None:
        traceback_and_raise(NotImplementedError)

    def register_in_memory_client(self, client: AbstractNodeClient) -> None:
        # WARNING: Gross hack
        route_index = self.default_route_index
        # this ID should be unique but persistent so that lookups are universal
        route = self.routes[route_index]
        if isinstance(route, SoloRoute):
            connection = route.connection
            if isinstance(connection, VirtualClientConnection):
                connection.server.node.in_memory_client_registry[
                    client.address.target_id.id
                ] = client
            else:
                traceback_and_raise(
                    Exception(
                        "Unable to save client reference without VirtualClientConnection"
                    )
                )
        else:
            traceback_and_raise(
                Exception("Unable to save client reference without SoloRoute")
            )

    def register(self, client: AbstractNodeClient) -> None:
        debug(f"> Registering {client.pprint} with {self.pprint}")
        self.register_in_memory_client(client=client)
        msg = RegisterChildNodeMessage(
            lookup_id=client.id,
            child_node_client_address=client.address,
            address=self.address,
        )

        if self.network is not None:
            client.network = (
                self.network if self.network is not None else client.network
            )

        # QUESTION
        # if the client is a network and the domain is not none this will set it
        # on the network causing an exception
        # but we can't check if the client is a NetworkClient here because
        # this is a superclass of NetworkClient
        # Remove: if self.domain is not None:
        # then see the test line node_test.py:
        # bob_network_client.register(client=bob_domain_client)
        if self.domain is not None:
            client.domain = self.domain if self.domain is not None else client.domain

        if self.device is not None:
            client.device = self.device if self.device is not None else client.device

            if self.device != client.device:
                raise AttributeError("Devices don't match")

        if self.vm is not None:
            client.vm = self.vm

        self.send_immediate_msg_without_reply(msg=msg)

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        traceback_and_raise(NotImplementedError)

    # TODO fix the msg type but currently tensor needs SyftMessage

    def send_immediate_msg_with_reply(
        self,
        msg: Union[SignedImmediateSyftMessageWithReply, ImmediateSyftMessageWithReply],
        route_index: int = 0,
    ) -> SyftMessage:
        route_index = route_index or self.default_route_index

        if isinstance(msg, ImmediateSyftMessageWithReply):
            output = (
                f"> {self.pprint} Signing {msg.pprint} with "
                + f"{self.key_emoji(key=self.signing_key.verify_key)}"
            )
            debug(output)
            msg = msg.sign(signing_key=self.signing_key)

        response = self.routes[route_index].send_immediate_msg_with_reply(msg=msg)
        if response.is_valid:
            # check if we have an ExceptionMessage to trigger a local exception
            # from a remote exception that we caused
            if isinstance(response.message, ExceptionMessage):
                exception_msg = response.message
                exception = exception_msg.exception_type(exception_msg.exception_msg)
                error(str(exception))
                traceback_and_raise(exception)
            else:
                return response.message

        traceback_and_raise(
            Exception("Response was signed by a fake key or was corrupted in transit.")
        )

    # TODO fix the msg type but currently tensor needs SyftMessage

    def send_immediate_msg_without_reply(
        self,
        msg: Union[
            SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply
        ],
        route_index: int = 0,
    ) -> None:
        route_index = route_index or self.default_route_index

        if isinstance(msg, ImmediateSyftMessageWithoutReply):
            output = (
                f"> {self.pprint} Signing {msg.pprint} with "
                + f"{self.key_emoji(key=self.signing_key.verify_key)}"
            )
            debug(output)
            msg = msg.sign(signing_key=self.signing_key)
        debug(f"> Sending {msg.pprint} {self.pprint} ➡️  {msg.address.pprint}")
        self.routes[route_index].send_immediate_msg_without_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply, route_index: int = 0
    ) -> None:
        route_index = route_index or self.default_route_index
        output = (
            f"> {self.pprint} Signing {msg.pprint} with "
            + f"{self.key_emoji(key=self.signing_key.verify_key)}"
        )
        debug(output)
        signed_msg: SignedEventualSyftMessageWithoutReply = msg.sign(
            signing_key=self.signing_key
        )

        self.routes[route_index].send_eventual_msg_without_reply(msg=signed_msg)

    def __repr__(self) -> str:
        return f"<Client pointing to node with id:{self.id}>"

    def register_route(self, route: Route) -> None:
        self.routes.append(route)

    def set_default_route(self, route_index: int) -> None:
        self.default_route = route_index

    def _object2proto(self) -> Client_PB:
        obj_type = get_fully_qualified_name(obj=self)

        routes = [serialize(route) for route in self.routes]

        network = self.network._object2proto() if self.network is not None else None

        domain = self.domain._object2proto() if self.domain is not None else None

        device = self.device._object2proto() if self.device is not None else None

        vm = self.vm._object2proto() if self.vm is not None else None

        client_pb = Client_PB(
            obj_type=obj_type,
            id=serialize(self.id),
            name=self.name,
            routes=routes,
            has_network=self.network is not None,
            network=network,
            has_domain=self.domain is not None,
            domain=domain,
            has_device=self.device is not None,
            device=device,
            has_vm=self.vm is not None,
            vm=vm,
        )

        return client_pb

    @staticmethod
    def _proto2object(proto: Client_PB) -> "Client":
        module_parts = proto.obj_type.split(".")
        klass = module_parts.pop()
        obj_type = getattr(sys.modules[".".join(module_parts)], klass)

        network = (
            SpecificLocation._proto2object(proto.network) if proto.has_network else None
        )
        domain = (
            SpecificLocation._proto2object(proto.domain) if proto.has_domain else None
        )
        device = (
            SpecificLocation._proto2object(proto.device) if proto.has_device else None
        )
        vm = SpecificLocation._proto2object(proto.vm) if proto.has_vm else None
        routes = [SoloRoute._proto2object(route) for route in proto.routes]

        obj = obj_type(
            name=proto.name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
        )

        if type(obj) != obj_type:
            traceback_and_raise(
                TypeError(
                    f"Deserializing Client. Expected type {obj_type}. Got {type(obj)}"
                )
            )

        return obj

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Client_PB

    @property
    def keys(self) -> str:
        verify = (
            self.key_emoji(key=self.signing_key.verify_key)
            if self.signing_key is not None
            else "🚫"
        )
        keys = f"🔑 {verify}"

        return keys

    def __hash__(self) -> Any:
        return hash(self.id)


class StoreClient:
    def __init__(self, client: Client) -> None:
        self.client = client

    @property
    def store(self) -> List[Pointer]:
        msg = ObjectSearchMessage(
            address=self.client.address, reply_to=self.client.address
        )

        results = getattr(
            self.client.send_immediate_msg_with_reply(msg=msg), "results", None
        )
        if results is None:
            traceback_and_raise(ValueError("TODO"))

        # This is because of a current limitation in Pointer where we cannot
        # serialize a client object. TODO: Fix limitation in Pointer so that we don't need this.
        for result in results:
            result.gc_enabled = False
            result.client = self.client

        return results

    def __len__(self) -> int:
        """Return the number of items in the object store we're allowed to know about"""

        return len(self.store)

    def __getitem__(self, key: Union[str, int]) -> Pointer:
        if isinstance(key, str):
            matches = 0
            match_obj: Optional[Pointer] = None

            for obj in self.store:
                if key in obj.tags:
                    matches += 1
                    match_obj = obj
            if matches == 1 and match_obj is not None:
                return match_obj
            elif matches > 1:
                traceback_and_raise(KeyError("More than one item with tag:" + str(key)))
            else:
                # If key does not math with any tags, we then try to match it with id string.
                # But we only do this if len(key)>=5, because if key is too short, for example
                # if key="a", there are chances of mismatch it with id string, and I don't
                # think the user pass a key such short as part of id string.
                if len(key) >= 5:
                    for obj in self.store:
                        if key in str(obj.id_at_location.value).replace("-", ""):
                            return obj
                else:
                    traceback_and_raise(
                        KeyError(
                            f"No such item found for tag: {key}, and we "
                            + "don't consider it as part of id string because its too short."
                        )
                    )

            traceback_and_raise(KeyError("No such item found for id:" + str(key)))
        if isinstance(key, int):
            return self.store[key]
        else:
            traceback_and_raise(KeyError("Please pass in a string or int key"))

    def __repr__(self) -> str:
        return repr(self.store)

    @property
    def pandas(self) -> pd.DataFrame:
        obj_lines: List[Dict[str, Any]] = list()
        for obj in self.store:
            obj_lines.append(
                {
                    "ID": obj.id_at_location,
                    "Tags": obj.tags,
                    "Description": obj.description,
                    "object_type": obj.object_type,
                }
            )
        return pd.DataFrame(obj_lines)
