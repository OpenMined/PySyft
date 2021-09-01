# relative
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.serde.serializable import Serializable
from ...logger import traceback_and_raise


class BidirectionalConnection(Serializable):
    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)


class ServerConnection(Serializable):
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)


class ClientConnection(Serializable):
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()

    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        traceback_and_raise(NotImplementedError)
