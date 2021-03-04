# stdlib
from copy import copy
from typing import Any
from typing import Callable
from typing import Type

# syft relative
from ....core.common.serde.deserialize import _deserialize
from ....proto.core.io.address_pb2 import Address as Address_PB
from ...core.node.common.client import Client
from ...core.pointer.pointer import Pointer
from ...messages.infra_messages import CreateWorkerMessage
from ...messages.infra_messages import DeleteWorkerMessage
from ...messages.infra_messages import GetWorkerMessage
from ...messages.infra_messages import GetWorkersMessage
from ...messages.infra_messages import UpdateWorkerMessage
from ...messages.transfer_messages import SaveObjectMessage
from .request_api import GridRequestAPI


class WorkerRequestAPI(GridRequestAPI):
    response_key = "worker"

    def __init__(self, send: Callable, client: Client):
        super().__init__(
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            send=send,
            response_key=WorkerRequestAPI.response_key,
        )
        self.__client = client

    def __getitem__(self, key: int) -> object:
        return self.get(worker_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(worker_id=key)

    def to_obj(self, result: Any) -> Any:
        _raw_worker = super().to_obj(result)
        _raw_addr = _raw_worker.address.encode("ISO-8859-1")

        addr_pb = Address_PB()
        addr_pb.ParseFromString(_raw_addr)

        _worker_obj = copy(self.__client)
        _worker_obj.proxy_address = _deserialize(blob=addr_pb)

        def _save(obj_ptr: Type[Pointer]) -> None:
            _content = {
                "uid": str(obj_ptr.id_at_location.value),
            }
            return _worker_obj.__perform_grid_request(
                grid_msg=SaveObjectMessage, content=_content
            )

        return _worker_obj
