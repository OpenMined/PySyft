# stdlib
from typing import Any
from typing import Dict

# third party
from pandas import DataFrame

# syft relative
from ...messages.infra_messages import CreateWorkerMessage
from ...messages.infra_messages import DeleteWorkerMessage
from ...messages.infra_messages import GetWorkerMessage
from ...messages.infra_messages import GetWorkersMessage
from ...messages.infra_messages import UpdateWorkerMessage
from .request_api import GridRequestAPI


class WorkerRequestAPI(GridRequestAPI):
    response_key = "worker"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            send=send,
            response_key=WorkerRequestAPI.response_key,
        )

    def __getitem__(self, key):
        return self.get(worker_id=key)

    def __delitem__(self, key):
        self.delete(worker_id=key)
