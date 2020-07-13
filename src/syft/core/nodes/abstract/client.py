from ...message.syft_message import SyftMessage
from ....typecheck import type_hints
from ....common.id import UID
from ...io.abstract import ClientConnection


class Client:

    @type_hints
    def __init__(self, worker_id: UID, connection: ClientConnection):
        self.worker_id = worker_id
        self.connection = connection

    @type_hints
    def send_msg(self, msg:SyftMessage) -> SyftMessage:
        return self.connection.send_msg(msg=msg)

    @type_hints
    def __repr__(self) -> str:
        return f"<Client pointing to worker with id:{self.worker_id}>"
