from ...message.syft_message import SyftMessage
from ....decorators import syft_decorator
from ....common.id import UID
from ...io.abstract import ClientConnection

class Client:
    @syft_decorator(typechecking=True)
    def __init__(self, worker_id: UID, name: str, connection: ClientConnection):
        self.worker_id = worker_id
        self.name = name
        self.connection = connection

    @syft_decorator(typechecking=True)
    def send_msg(self, msg: SyftMessage) -> SyftMessage:
        return self.connection.send_msg(msg=msg)

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        return f"<Client pointing to worker with id:{self.worker_id}>"
