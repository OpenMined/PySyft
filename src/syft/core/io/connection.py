from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
    SignedMessage,
)

from ...decorators import syft_decorator


class BidirectionalConnection(object):
    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    def recv_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    def recv_signed_msg_without_reply(self, msg: SignedMessage) -> SignedMessage:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_signed_msg_with_reply(self, msg: SignedMessage) -> SignedMessage:
        raise NotImplementedError


class ServerConnection(BidirectionalConnection):
    pass


class ClientConnection(BidirectionalConnection):
    pass
