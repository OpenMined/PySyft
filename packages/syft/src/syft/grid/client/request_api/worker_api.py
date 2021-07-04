# stdlib
from typing import Any
from typing import Callable
from typing import Type

# third party
from pandas import DataFrame

# relative
from ....core.common.serde.deserialize import _deserialize
from ....core.node.common.node import Node
from ....core.node.domain.enums import PyGridClientEnums
from ....core.node.domain.enums import RequestAPIFields
from ....core.node.domain.enums import ResponseObjectEnum
from ....core.node.domain.messages.transfer_messages import SaveObjectMessage
from ....core.pointer.pointer import Pointer
from ....proto.core.io.address_pb2 import Address as Address_PB
from ...messages.infra_messages import CreateWorkerMessage
from ...messages.infra_messages import DeleteWorkerMessage
from ...messages.infra_messages import GetWorkerInstanceTypesMessage
from ...messages.infra_messages import GetWorkerMessage
from ...messages.infra_messages import GetWorkersMessage
from ...messages.infra_messages import UpdateWorkerMessage
from .request_api import GridRequestAPI


class WorkerRequestAPI(GridRequestAPI):
    def __init__(self, node: Type[Node]):
        super().__init__(
            node=node,
            create_msg=CreateWorkerMessage,
            get_msg=GetWorkerMessage,
            get_all_msg=GetWorkersMessage,
            update_msg=UpdateWorkerMessage,
            delete_msg=DeleteWorkerMessage,
            response_key=ResponseObjectEnum.WORKER,
        )

    def instance_type(self, pandas: bool = False) -> Any:
        result = self.perform_api_request(grid_msg=GetWorkerInstanceTypesMessage)
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

        _worker_obj = self.node.__class__(
            credentials={},
            url=self.node.conn.base_url,
            conn_type=self.node.conn.__class__,
            client_type=self.node.client_type,
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
                    RequestAPIFields.DOMAIN_ADDRESS: self.node.conn.base_url,
                },
            }
            signed_msg = SaveObjectMessage(
                address=_content[RequestAPIFields.ADDRESS],
                content=_content[RequestAPIFields.CONTENT],
            ).sign(signing_key=_worker_obj.signing_key)
            _worker_obj.send_immediate_msg_without_reply(msg=signed_msg)

        _worker_obj.save = _save
        return _worker_obj
