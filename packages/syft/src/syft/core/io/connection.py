# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...logger import traceback_and_raise


class BidirectionalConnection(object):
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

    def _object2proto(self) -> None:
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> "BidirectionalConnection":
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        traceback_and_raise(NotImplementedError)


class ServerConnection(object):
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

    def _object2proto(self) -> None:
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> "ServerConnection":
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        traceback_and_raise(NotImplementedError)


class ClientConnection(object):
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

    def _object2proto(self) -> None:
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> object:
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        traceback_and_raise(NotImplementedError)
