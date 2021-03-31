# stdlib
from typing import Any
from typing import Callable
from typing import Type

# third party
from pandas import DataFrame

# syft relative
from ....core.common.serde.deserialize import _deserialize
from ....core.node.common.client import Client
from ....core.pointer.pointer import Pointer
from ....proto.core.io.address_pb2 import Address as Address_PB
from ...messages.infra_messages import CreateWorkerMessage
from ...messages.infra_messages import DeleteWorkerMessage
from ...messages.infra_messages import GetWorkerInstanceTypesMessage
from ...messages.infra_messages import GetWorkerMessage
from ...messages.infra_messages import GetWorkersMessage
from ...messages.infra_messages import UpdateWorkerMessage
from ...messages.transfer_messages import SaveObjectMessage
from .request_api import GridRequestAPI


class WorkerRequestAPI(GridRequestAPI):
    response_key = "worker"

    def __init__(self, send: Callable, domain_client: Client):
        super().__init__(
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            send=send,
            response_key=WorkerRequestAPI.response_key,
        )

        self.domain_client = domain_client

    def instance_type(self, pandas: bool = False) -> Any:
        result = self.__send(grid_msg=GetWorkerInstanceTypesMessage)
        if pandas:
            result = DataFrame(result)
        return result

    def __getitem__(self, key: int) -> object:
        return self.get(worker_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(worker_id=key)

    def to_obj(self, result: Any) -> Any:
        _raw_worker = super().to_obj(result)
        _raw_addr = _raw_worker.syft_address.encode("ISO-8859-1")

        addr_pb = Address_PB()
        addr_pb.ParseFromString(_raw_addr)

        _worker_obj = self.domain_client.__class__(  # type: ignore
            credentials={},
            url=self.domain_client.conn.base_url,  # type: ignore
            conn_type=self.domain_client.conn.__class__,  # type: ignore
            client_type=self.domain_client.client_type,  # type: ignore
        )
        _worker_obj.proxy_address = _deserialize(blob=addr_pb)  # type: ignore
        _worker_obj.domain = _worker_obj.proxy_address.domain  # type: ignore

        for key, value in result.items():
            try:
                setattr(_worker_obj, key, value)
            except AttributeError:
                continue

        def _save(obj_ptr: Type[Pointer]) -> None:
            _content = {
                "address": _worker_obj.address,
                "content": {"uid": str(obj_ptr.id_at_location.value)},
            }
            signed_msg = SaveObjectMessage(**_content).sign(  # type: ignore
                signing_key=_worker_obj.signing_key
            )
            _worker_obj.send_immediate_msg_without_reply(msg=signed_msg)

        _worker_obj.save = _save  # type: ignore
        return _worker_obj
