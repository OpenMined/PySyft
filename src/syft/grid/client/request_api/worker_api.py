# stdlib
from typing import Callable

# syft relative
from ...messages.infra_messages import CreateWorkerMessage
from ...messages.infra_messages import DeleteWorkerMessage
from ...messages.infra_messages import GetWorkerMessage
from ...messages.infra_messages import GetWorkersMessage
from ...messages.infra_messages import UpdateWorkerMessage
from .request_api import GridRequestAPI


class WorkerRequestAPI(GridRequestAPI):
    response_key = "worker"

    def __init__(self, send: Callable):
        super().__init__(
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            send=send,
            response_key=WorkerRequestAPI.response_key,
        )

    def __getitem__(self, key: int) -> object:
        return self.get(worker_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(worker_id=key)
