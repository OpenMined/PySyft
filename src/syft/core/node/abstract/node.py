# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft relative
from ....decorators import syft_decorator
from ...common.message import SignedEventualSyftMessageWithoutReply
from ...common.message import SignedImmediateSyftMessageWithReply
from ...common.message import SignedImmediateSyftMessageWithoutReply
from ...common.uid import UID
from ...io.address import Address
from ...io.location import Location
from ...store import ObjectStore


class AbstractNode(Address):

    name: Optional[str]
    signing_key: Optional[SigningKey]
    verify_key: Optional[VerifyKey]
    root_verify_key: VerifyKey
    guest_verify_key_registry: Set[VerifyKey]
    admin_verify_key_registry: Set[VerifyKey]
    cpl_ofcr_verify_key_registry: Set[VerifyKey]

    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry: Dict[Any, Any]
    # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
    signaling_msgs: Dict[Any, Any]

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
    lib_ast: Any  # Can't import Globals (circular reference)
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
    lib_ast: Any  # Can't import Globals (circular reference)
    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry: Dict[Any, Any]
    """"""

    @property
    def id(self) -> UID:
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError
