from typing import Any, List, Optional
from typing import Set
from typing import Dict

from nacl.signing import SigningKey
from nacl.signing import VerifyKey

from ...common.uid import UID
from ...io.address import Address
from syft.core.io.location import Location
from syft.decorators import syft_decorator
from ...store import ObjectStore
from ...common.message import (
    SignedImmediateSyftMessageWithoutReply,
    SignedEventualSyftMessageWithoutReply,
    SignedImmediateSyftMessageWithReply,
)


class AbstractNode(Address):

    name: Optional[str]
    signing_key: Optional[SigningKey]
    verify_key: Optional[VerifyKey]
    root_verify_key: VerifyKey
    guest_verify_key_registry: Set[VerifyKey]

    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry: Dict[Any, Any]

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

    store: ObjectStore
    requests: List
    lib_ast: Any  # Cant import Globals (circular reference)
    """"""

    @property
    def known_child_nodes(self) -> List[Any]:
        raise NotImplementedError

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError

    @property
    def keys(self) -> str:
        verify = (
            self.key_emoji(key=self.signing_key.verify_key)
            if self.signing_key is not None
            else "🚫"
        )
        root = (
            self.key_emoji(key=self.root_verify_key)
            if self.root_verify_key is not None
            else "🚫"
        )
        keys = f"🔑 {verify}" + f"🗝 {root}"

        return keys


class AbstractNodeClient(Address):
    lib_ast: Any  # Cant import Globals (circular reference)
    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry: Dict[Any, Any]
    """"""

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError
