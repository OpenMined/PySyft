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
from .service_request import GridServiceRequest


class WorkerServiceRequest(GridServiceRequest):
    response_key = "worker"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            send=send,
            response_key=WorkerServiceRequest.response_key,
        )
