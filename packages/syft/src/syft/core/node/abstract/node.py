# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import BaseSettings

# relative
from ....logger import traceback_and_raise
from ....util import key_emoji
from ...common.message import SignedImmediateSyftMessageWithReply
from ...common.message import SignedImmediateSyftMessageWithoutReply
from ...common.uid import UID
from ...io.address import Address
from ...io.location import Location
from ...store import ObjectStore


class AbstractNodeClient(Address):
    lib_ast: Any  # Can't import Globals (circular reference)
    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry: Dict[Any, Any]
    signing_key: SigningKey
    verify_key: VerifyKey
    node_uid: UID
    """"""

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        traceback_and_raise(NotImplementedError)

    def send_immediate_msg_without_reply(self, msg: Any) -> Any:
        raise NotImplementedError

    def send_immediate_msg_with_reply(
        self, msg: Any, timeout: Optional[float] = None, return_signed: bool = False
    ) -> Any:
        raise NotImplementedError


class AbstractNode(Address):
    name: Optional[str]
    signing_key: Optional[SigningKey]
    guest_signing_key_registry: Set[SigningKey]
    guest_verify_key_registry: Set[VerifyKey]
    admin_verify_key_registry: Set[VerifyKey]
    cpl_ofcr_verify_key_registry: Set[VerifyKey]
    acc: Optional[Any]
    settings: BaseSettings
    node_uid: UID

    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry: Dict[Any, Any]
    # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
    signaling_msgs: Dict[Any, Any]

    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        settings: Optional[BaseSettings] = None,
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )
        self.settings = settings

    store: ObjectStore
    requests: List
    lib_ast: Any  # Can't import Globals (circular reference)
    """"""

    @property
    def known_child_nodes(self) -> List[Any]:
        traceback_and_raise(NotImplementedError)

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        traceback_and_raise(NotImplementedError)

    @property
    def icon(self) -> str:
        icon = "🏰"
        return icon

    @property
    def pprint(self) -> str:
        output = f"{self.icon} {self.name} ({str(self.__class__.__name__)})"
        if hasattr(self, "id"):
            _id = getattr(self, "node_uid", None)
            emoji = getattr(_id, "emoji", None)
            emoji_str = emoji() if emoji is not None else None
            output += f"@{emoji_str}"
        return output

    @property
    def keys(self) -> str:
        verify = (
            key_emoji(key=self.signing_key.verify_key)
            if self.signing_key is not None
            else "🚫"
        )
        root = (
            key_emoji(key=self.root_verify_key)
            if self.root_verify_key is not None
            else "🚫"
        )
        keys = f"🔑 {verify}" + f"🗝 {root}"

        return keys

    def get_peer_client(
        self, node_id: UID, only_vpn: bool = True
    ) -> Optional[AbstractNodeClient]:
        traceback_and_raise(NotImplementedError)

    def __repr__(self) -> str:
        out = f"<{type(self).__name__} -"
        return out[:-1] + ">"
