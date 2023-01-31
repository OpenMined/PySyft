# stdlib
from typing import Callable
from typing import Union

# third party
from nacl.bindings.crypto_sign import crypto_sign_keypair
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID
from syft.core.node.common.node import Node
from syft.core.node.common.node_manager.dict_store import DictStore


class MockNode(Node):
    """Mock Node object for testing purposes."""

    # use a dict store for these tests
    def __init__(
        self,
    ):
        super().__init__(store_type=DictStore)

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        return None

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        return construct_dummy_message(SignedImmediateSyftMessageWithoutReply)


def construct_dummy_message(
    msg_class: Callable,
) -> Union[
    SignedImmediateSyftMessageWithReply,
    SignedImmediateSyftMessageWithoutReply,
]:
    """
    Helper method to construct a dummy SignedMessage of the following:
    - SignedImmediateSyftMessageWithReply
    - SignedImmediateSyftMessageWithoutReply
    """
    address = UID()
    key = VerifyKey(crypto_sign_keypair()[0])

    return msg_class(
        address=address,
        signature=b"my_signature",
        verify_key=key,
        message=b"hello_world",
    )
