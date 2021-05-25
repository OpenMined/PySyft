# stdlib
from typing import Callable
from typing import Union

# third party
from nacl.bindings.crypto_sign import crypto_sign_keypair
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import SignedEventualSyftMessageWithoutReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location.specific import SpecificLocation
from syft.core.node.common.node import Node


class MockNode(Node):
    """Mock Node object for testing purposes."""

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        return None

    def recv_eventual_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> None:
        return None

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        return construct_dummy_message(SignedImmediateSyftMessageWithoutReply)


def _construct_address() -> Address:
    """Helper method to construct an Address"""
    return Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )


def construct_dummy_message(
    msg_class: Callable,
) -> Union[
    SignedEventualSyftMessageWithoutReply,
    SignedImmediateSyftMessageWithReply,
    SignedImmediateSyftMessageWithoutReply,
]:
    """
    Helper method to construct a dummy SignedMessage of the following:
    - SignedEventualSyftMessageWithoutReply
    - SignedImmediateSyftMessageWithReply
    - SignedImmediateSyftMessageWithoutReply
    """
    address = _construct_address()
    key = VerifyKey(crypto_sign_keypair()[0])

    return msg_class(
        address=address,
        obj_type="signed_message",
        signature=b"my_signature",
        verify_key=key,
        message=b"hello_world",
    )
