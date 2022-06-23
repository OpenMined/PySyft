# stdlib
from typing import Optional

# relative
from ...logger import traceback_and_raise
from ..common.message import SignedEventualSyftMessageWithoutReply
from ..common.message import SignedImmediateSyftMessageWithReply
from ..common.message import SignedImmediateSyftMessageWithoutReply


class BidirectionalConnection:
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
        self,
        msg: SignedImmediateSyftMessageWithReply,
        timeout: Optional[float] = None,
        return_signed: bool = False,
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    def send_immediate_msg_without_reply(
        self,
        msg: SignedImmediateSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def send_eventual_msg_without_reply(
        self,
        msg: SignedEventualSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        traceback_and_raise(NotImplementedError)


class ServerConnection:
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()

    def recv_immediate_msg_with_reply(
        self,
        msg: SignedImmediateSyftMessageWithReply,
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


class ClientConnection:
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()

    def send_immediate_msg_with_reply(
        self,
        msg: SignedImmediateSyftMessageWithReply,
        timeout: Optional[float] = None,
        return_signed: bool = False,
    ) -> SignedImmediateSyftMessageWithoutReply:
        traceback_and_raise(NotImplementedError)

    def send_immediate_msg_without_reply(
        self,
        msg: SignedImmediateSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        traceback_and_raise(NotImplementedError)

    def send_eventual_msg_without_reply(
        self,
        msg: SignedEventualSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        traceback_and_raise(NotImplementedError)
