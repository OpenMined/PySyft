# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...decorators import syft_decorator


class BidirectionalConnection(object):
    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> "BidirectionalConnection":
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        raise NotImplementedError


class ServerConnection(object):
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> "ServerConnection":
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        raise NotImplementedError


class ClientConnection(object):
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithReply:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> object:
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        raise NotImplementedError
