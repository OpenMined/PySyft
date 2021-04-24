# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Type

# third party
from pandas import DataFrame

# syft relative
from ....core.io.connection import ClientConnection
from ....core.node.domain.client import DomainClient
from ...messages.dataset_messages import CreateDatasetMessage
from ...messages.dataset_messages import DeleteDatasetMessage
from ...messages.dataset_messages import GetDatasetMessage
from ...messages.dataset_messages import GetDatasetsMessage
from ...messages.dataset_messages import UpdateDatasetMessage
from ..enums import ResponseObjectEnum
from .request_api import GridRequestAPI


class DatasetRequestAPI(GridRequestAPI):
    def __init__(
        self, send: Callable, conn: Type[ClientConnection], client: Type[DomainClient]
    ):
        super().__init__(
            create_msg=CreateDatasetMessage,
            get_msg=GetDatasetMessage,
            get_all_msg=GetDatasetsMessage,
            update_msg=UpdateDatasetMessage,
            delete_msg=DeleteDatasetMessage,
            send=send,
            response_key=ResponseObjectEnum.DATASET,
        )
        self.conn = conn
        self.client = client

    def create(self, path: str) -> Dict[str, str]:
        response = self.conn.send_files(path)
        return self.to_obj(response)

    def __getitem__(self, key: str) -> Any:
        return self.get(dataset_id=key)

    def __delitem__(self, key: str) -> Any:
        self.delete(dataset_id=key)

    def to_obj(self, result: Any) -> Any:
        dataset_obj = super().to_obj(result)
        dataset_obj.pandas = DataFrame(dataset_obj.data)
        datasets = []

        pointers = self.client.store
        for data in dataset_obj.data:
            _class_name = ResponseObjectEnum.DATA.capitalize()
            data_obj = type(_class_name, (object,), data)()
            data_obj.shape = eval(data_obj.shape)
            data_obj.pointer = pointers[data_obj.id.replace("-", "")]
            datasets.append(data_obj)

        dataset_obj.files = datasets
        type(dataset_obj).__getitem__ = lambda x, i: x.data[i]
        return dataset_obj
