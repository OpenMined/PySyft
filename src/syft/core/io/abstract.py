from ..message.syft_message import SyftMessage
from ...decorators import syft_decorator
from typing import final


@final
class ServerConnection(object):
    @syft_decorator(typechecking=True)
    def recv_msg(self, msg: SyftMessage) -> SyftMessage:
        raise NotImplementedError


@final
class ClientConnection(object):
    @syft_decorator(typechecking=True)
    def send_msg(self, msg: SyftMessage) -> SyftMessage:
        raise NotImplementedError
