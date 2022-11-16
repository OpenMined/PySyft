# stdlib
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pandas as pd

# syft absolute
import syft as sy

# relative
from ....grid import GridURL
from ....logger import critical
from ....logger import debug
from ....logger import error
from ....logger import info
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
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ...io.route import Route
from ...pointer.garbage_collection import GarbageCollection
from ...pointer.garbage_collection import gc_get_default_strategy
from ...pointer.pointer import Pointer
from ..abstract.node import AbstractNodeClient
from ..common.client_manager.node_networking_api import NodeNetworkingAPI
from .action.exception_action import ExceptionMessage
from .node_service.object_search.obj_search_service import ObjectSearchMessage


@serializable()
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
        version: Optional[str] = None,
    ):
        name = f"{name}" if name is not None else None
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
        self.networking = NodeNetworkingAPI(client=self)
        self.version = version

    def obj_exists(self, obj_id: UID) -> bool:
        raise NotImplementedError

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
        meta = sy.deserialize(blob=metadata)
        return meta.node, meta.name, meta.id

    def install_supported_frameworks(self) -> None:
        self.lib_ast = sy.lib.create_lib_ast(client=self)

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
                    self.python = python_attr
                    python_attr = getattr(lib_attr, "adp", None)
                    self.adp = python_attr

            except Exception as e:
                critical(f"Failed to set python attribute on client. {e}")

    def configure(self, **kwargs: Any) -> Any:
        # relative
        from .node_service.node_setup.node_setup_messages import UpdateSetupMessage

        if "daa_document" in kwargs.keys():
            kwargs["daa_document"] = open(kwargs["daa_document"], "rb").read()
        else:
            kwargs["daa_document"] = b""
        response = self._perform_grid_request(  # type: ignore
            grid_msg=UpdateSetupMessage, content=kwargs
        ).content
        info(response)

    @property
    def settings(self, **kwargs: Any) -> Dict[Any, Any]:  # type: ignore
        try:
            # relative
            from .node_service.node_setup.node_setup_messages import GetSetUpMessage

            return self._perform_grid_request(  # type: ignore
                grid_msg=GetSetUpMessage, content=kwargs
            ).content  # type : ignore
        except Exception:  # nosec
            # unable to fetch settings
            return {}

    def join_network(
        self,
        client: Optional[AbstractNodeClient] = None,
        host_or_ip: Optional[str] = None,
    ) -> None:
        # this asks for a VPN key so it must be on a public interface hence the
        # client or a public host_or_ip
        try:
            if client is None and host_or_ip is None:
                raise ValueError(
                    "join_network requires a Client object or host_or_ip string"
                )

            # we are leaving the client and entering the node in a container
            # any hostnames of localhost need to be converted to docker-host
            if client is not None:
                grid_url = client.routes[0].connection.base_url  # type: ignore
            else:
                grid_url = GridURL.from_url(str(host_or_ip))

            return self.vpn.join_network_vpn(grid_url=grid_url)  # type: ignore
        except Exception as e:
            msg = f"Failed to join network with {client} or {host_or_ip}. {e}"
            raise Exception(msg)

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        traceback_and_raise(NotImplementedError)

    # TODO fix the msg type but currently tensor needs SyftMessage

    def send_immediate_msg_with_reply(
        self,
        msg: Union[
            SignedImmediateSyftMessageWithReply,
            ImmediateSyftMessageWithReply,
            Any,  # TEMPORARY until we switch everything to NodeRunnableMessage types.
        ],
        timeout: Optional[float] = None,
        return_signed: bool = False,
        route_index: int = 0,
    ) -> Union[SyftMessage, SignedMessage]:

        # relative
        from .node_service.simple.simple_messages import NodeRunnableMessageWithReply
        from .node_service.tff.tff_messages import TFFMessageWithReply

        # TEMPORARY: if message is instance of NodeRunnableMessageWithReply then we need to wrap it in a SimpleMessage
        if isinstance(msg, NodeRunnableMessageWithReply) or isinstance(
            msg, TFFMessageWithReply
        ):
            msg = msg.prepare(address=self.address, reply_to=self.address)

        route_index = route_index or self.default_route_index

        if isinstance(msg, ImmediateSyftMessageWithReply):
            output = (
                f"> {self.pprint} Signing {msg.pprint} with "
                + f"{self.key_emoji(key=self.signing_key.verify_key)}"
            )
            debug(output)
            msg = msg.sign(signing_key=self.signing_key)

        response = self.routes[route_index].send_immediate_msg_with_reply(
            msg=msg, timeout=timeout
        )
        if response.is_valid:
            # check if we have an ExceptionMessage to trigger a local exception
            # from a remote exception that we caused
            if isinstance(response.message, ExceptionMessage):
                exception_msg = response.message
                exception = exception_msg.exception_type(exception_msg.exception_msg)
                error(str(exception))
                traceback_and_raise(exception)
            else:
                if return_signed:
                    return response
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
        timeout: Optional[float] = None,
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
        self.routes[route_index].send_immediate_msg_without_reply(
            msg=msg, timeout=timeout
        )

    def send_eventual_msg_without_reply(
        self,
        msg: EventualSyftMessageWithoutReply,
        route_index: int = 0,
        timeout: Optional[float] = None,
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

        self.routes[route_index].send_eventual_msg_without_reply(
            msg=signed_msg, timeout=timeout
        )

    def url_from_path(self, path: str) -> str:
        new_url = GridURL.from_url(url=path)
        client_url = self.routes[0].connection.base_url.copy()  # type: ignore
        new_url.protocol = client_url.protocol
        new_url.port = client_url.port
        new_url.host_or_ip = client_url.host_or_ip
        return new_url.url

    def __repr__(self) -> str:
        return f"<Client pointing to node with id:{self.id}>"

    def register_route(self, route: Route) -> None:
        self.routes.append(route)

    def set_default_route(self, route_index: int) -> None:
        self.default_route = route_index

    def _object2proto(self) -> Client_PB:
        client_pb = Client_PB(
            obj_type=get_fully_qualified_name(obj=self),
            id=sy.serialize(self.id),
            name=self.name,
            routes=[sy.serialize(route) for route in self.routes],
            network=self.network._object2proto() if self.network else None,
            domain=self.domain._object2proto() if self.domain else None,
            device=self.device._object2proto() if self.device else None,
            vm=self.vm._object2proto() if self.vm else None,
        )
        return client_pb

    @staticmethod
    def _proto2object(proto: Client_PB) -> "Client":
        module_parts = proto.obj_type.split(".")
        klass = module_parts.pop()
        obj_type = getattr(sys.modules[".".join(module_parts)], klass)

        obj = obj_type(
            name=proto.name,
            routes=[sy.deserialize(route) for route in proto.routes],
            network=sy.deserialize(proto.network)
            if proto.HasField("network")
            else None,
            domain=sy.deserialize(proto.domain) if proto.HasField("domain") else None,
            device=sy.deserialize(proto.device) if proto.HasField("device") else None,
            vm=sy.deserialize(proto.vm) if proto.HasField("vm") else None,
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


GET_OBJECT_TIMEOUT = 60  # seconds


class StoreClient:
    def __init__(self, client: Client) -> None:
        self.client = client

    @property
    def store(self) -> List[Pointer]:
        msg = ObjectSearchMessage(
            address=self.client.address, reply_to=self.client.address
        )

        results = getattr(
            self.client.send_immediate_msg_with_reply(
                msg=msg, timeout=GET_OBJECT_TIMEOUT
            ),
            "results",
            None,
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

    def __iter__(self) -> Iterator[Any]:
        return self.store.__iter__()

    def __getitem__(self, key: Union[str, int, UID]) -> Pointer:
        return self.get(key=key)

    #
    # def __getitem__(self, key: Union[str, int, UID]) -> Pointer:
    #
    #     if isinstance(key, int):
    #         return self.store[key]
    #     elif isinstance(key, str):
    #         # PART 1: try using the key as an ID
    #         try:
    #             key = UID.from_string(key)
    #             return self[key]
    #         except ValueError:
    #
    #             # If there's no id of this key, then try matching on a tag
    #             matches = 0
    #             match_obj: Optional[Pointer] = None
    #
    #             for obj in self.store:
    #                 if key in obj.tags:
    #                     matches += 1
    #                     match_obj = obj
    #             if matches == 1 and match_obj is not None:
    #                 return match_obj
    #             else:  # matches > 1
    #                 traceback_and_raise(
    #                     KeyError("More than one item with tag:" + str(key))
    #                 )
    #                 raise KeyError("More than one item with tag:" + str(key))
    #
    #     elif isinstance(key, UID):
    #         msg = ObjectSearchMessage(
    #             address=self.client.address, reply_to=self.client.address, obj_id=key
    #         )
    #
    #         results = getattr(
    #             self.client.send_immediate_msg_with_reply(msg=msg), "results", None
    #         )
    #         if results is None:
    #             traceback_and_raise(ValueError("TODO"))
    #
    #         # This is because of a current limitation in Pointer where we cannot
    #         # serialize a client object. TODO: Fix limitation in Pointer so that we don't need this.
    #         for result in results:
    #             result.gc_enabled = False
    #             result.client = self.client
    #         if len(results) == 1:
    #             return results[0]
    #         return results
    #     else:
    #         traceback_and_raise(KeyError("Please pass in a string or int key"))

    def get(self, key: Union[str, int, UID]) -> Pointer:
        if isinstance(key, str):
            try:
                return self.get(UID.from_string(key))
            except IndexError:
                matches = 0
                match_obj: Optional[Pointer] = None

                for obj in self.store:
                    if key in obj.tags:
                        matches += 1
                        match_obj = obj
                if matches == 1 and match_obj is not None:
                    return match_obj
                elif matches > 1:
                    traceback_and_raise(
                        KeyError("More than one item with tag:" + str(key))
                    )
                else:
                    # If key does not math with any tags, we then try to match it with id string.
                    # But we only do this if len(key)>=5, because if key is too short, for example
                    # if key="a", there are chances of mismatch it with id string, and I don't
                    # think the user pass a key such short as part of id string.
                    str_key = str(key)
                    if len(str_key) >= 5:
                        for obj in self.store:
                            if str_key in str(obj.id_at_location.value).replace(
                                "-", ""
                            ):
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
        elif isinstance(key, UID):
            msg = ObjectSearchMessage(
                address=self.client.address, reply_to=self.client.address, obj_id=key
            )
            results = getattr(
                self.client.send_immediate_msg_with_reply(
                    msg=msg, timeout=GET_OBJECT_TIMEOUT
                ),
                "results",
                None,
            )

            if results is None:
                traceback_and_raise(ValueError("TODO"))

            # This is because of a current limitation in Pointer where we cannot
            # serialize a client object. TODO: Fix limitation in Pointer so that we don't need this.
            for result in results:
                result.gc_enabled = False
                result.client = self.client

            return results[0]
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

    def _repr_html_(self) -> str:
        return self.pandas._repr_html_()
