# stdlib
from typing import Any
from typing import Callable
from typing import Type

# third party
from pandas import DataFrame

# syft relative
from ....core.common.serde.deserialize import _deserialize
from ....core.pointer.pointer import Pointer
from ....proto.core.io.address_pb2 import Address as Address_PB
from ...messages.infra_messages import CreateWorkerMessage
from ...messages.infra_messages import DeleteWorkerMessage
from ...messages.infra_messages import GetWorkerInstanceTypesMessage
from ...messages.infra_messages import GetWorkerMessage
from ...messages.infra_messages import GetWorkersMessage
from ...messages.infra_messages import UpdateWorkerMessage
from ...messages.transfer_messages import SaveObjectMessage
from ..enums import PyGridClientEnums
from ..enums import RequestAPIFields
from ..enums import ResponseObjectEnum
from .request_api import GridRequestAPI


class WorkerRequestAPI(GridRequestAPI):
    def __init__(self, send: Callable, domain_client: Any):
        super().__init__(
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            send=send,
            response_key=ResponseObjectEnum.WORKER,
        )

        self.domain_client = domain_client

    def instance_type(self, pandas: bool = False) -> Any:
        result = self.send(grid_msg=GetWorkerInstanceTypesMessage)
        if pandas:
            max_size = len(min(result.values()))  # Why min/max functions were switched?
            for key in result.keys():
                empty_cells = max_size - len(result[key])
                while empty_cells:
                    result[key].append("")
                    empty_cells -= 1
            result = DataFrame(result)
        return result

    def __getitem__(self, key: int) -> object:
        return self.get(worker_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(worker_id=key)

    def to_obj(self, result: Any) -> Any:
        _raw_worker = super().to_obj(result)
        _raw_addr = _raw_worker.syft_address.encode(PyGridClientEnums.ENCODING)

        addr_pb = Address_PB()
        addr_pb.ParseFromString(_raw_addr)

        _worker_obj = self.domain_client.__class__(
            credentials={},
            url=self.domain_client.conn.base_url,
            conn_type=self.domain_client.conn.__class__,
            client_type=self.domain_client.client_type,
        )
        _worker_obj.proxy_address = _deserialize(blob=addr_pb)
        _worker_obj.domain = _worker_obj.proxy_address.domain

        for key, value in result.items():
            try:
                setattr(_worker_obj, key, value)
            except AttributeError:
                continue

        def _save(obj_ptr: Type[Pointer]) -> None:
            _content = {
                RequestAPIFields.ADDRESS: _worker_obj.address,
                RequestAPIFields.CONTENT: {
                    RequestAPIFields.UID: str(obj_ptr.id_at_location.value),
                    RequestAPIFields.DOMAIN_ADDRESS: self.domain_client.conn.base_url,
                },
            }
            signed_msg = SaveObjectMessage(
                address=_content[RequestAPIFields.ADDRESS],
                content=_content[RequestAPIFields.CONTENT],
            ).sign(signing_key=_worker_obj.signing_key)
            _worker_obj.send_immediate_msg_without_reply(msg=signed_msg)

        _worker_obj.save = _save
        return _worker_obj
