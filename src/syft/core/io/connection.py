from syft.core.message import SyftMessageWithReply
from syft.core.message import SyftMessageWithoutReply
from ...decorators import syft_decorator
from typing import final


@final
class ServerConnection(object):
    @syft_decorator(typechecking=True)
    def recv_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        raise NotImplementedError

    def recv_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        raise NotImplementedError


@final
class ClientConnection(object):
    @syft_decorator(typechecking=True)
    def send_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def send_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        raise NotImplementedError
