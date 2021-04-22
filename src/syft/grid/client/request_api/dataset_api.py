# stdlib
from typing import Any
from typing import Callable
from typing import Dict

# syft relative
from ...messages.dataset_messages import CreateDatasetMessage
from ...messages.dataset_messages import DeleteDatasetMessage
from ...messages.dataset_messages import GetDatasetMessage
from ...messages.dataset_messages import GetDatasetsMessage
from ...messages.dataset_messages import UpdateDatasetMessage
from ..enums import ResponseObjectEnum
from ..exceptions import PyGridClientException
from .request_api import GridRequestAPI


class DatasetRequestAPI(GridRequestAPI):
    def __init__(self, send: Callable):
        super().__init__(
            create_msg=CreateDatasetMessage,
            get_msg=GetDatasetMessage,
            get_all_msg=GetDatasetsMessage,
            update_msg=UpdateDatasetMessage,
            delete_msg=DeleteDatasetMessage,
            send=send,
            response_key=ResponseObjectEnum.DATASET,
        )

    def create(self, **kwargs: Any) -> Dict[str, str]:
        raise PyGridClientException(
            "You can't upload a dataset using PySyft Client, this feature will be implemented soon!"
        )

    def __getitem__(self, key: str) -> Any:
        return self.get(dataset_id=key)

    def __delitem__(self, key: str) -> Any:
        self.delete(dataset_id=key)
