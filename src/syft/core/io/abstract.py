from ..message.syft_message import SyftMessage
from ...typecheck import type_hints
from typing import final


@final
class ServerConnection(object):
    @type_hints
    def recv_msg(self, msg: SyftMessage) -> SyftMessage:
        raise NotImplementedError


@final
class ClientConnection(object):
    @type_hints
    def send_msg(self, msg: SyftMessage) -> SyftMessage:
        raise NotImplementedError
