# stdlib
import ast
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import Type
from typing import Union

# third party
from pandas import DataFrame

# syft absolute
from syft.core.node.common.service.dataset_manager.dataset_manager_messages import (
    CreateDatasetMessage,
)
from syft.core.node.common.service.dataset_manager.dataset_manager_messages import (
    DeleteDatasetMessage,
)
from syft.core.node.common.service.dataset_manager.dataset_manager_messages import (
    GetDatasetMessage,
)
from syft.core.node.common.service.dataset_manager.dataset_manager_messages import (
    GetDatasetsMessage,
)
from syft.core.node.common.service.dataset_manager.dataset_manager_messages import (
    UpdateDatasetMessage,
)

# relative
from ....node.common.node import Node
from ....node.domain.enums import RequestAPIFields
from ....node.domain.enums import ResponseObjectEnum
from ...common.request_api.request_api import RequestAPI


class DatasetRequestAPI(RequestAPI):
    def __init__(self, node: Type[Node]):
        super().__init__(
            node=node,
            create_msg=CreateDatasetMessage,
            get_msg=GetDatasetMessage,
            get_all_msg=GetDatasetsMessage,
            update_msg=UpdateDatasetMessage,
            delete_msg=DeleteDatasetMessage,
            response_key=ResponseObjectEnum.DATASET,
        )

    def create(self, path: str, **kwargs) -> Dict[str, str]:  # type: ignore
        response = self.node.conn.send_files(path, metadata=kwargs)  # type: ignore
        logging.info(response[RequestAPIFields.MESSAGE])

    def __getitem__(self, key: Union[str, int, slice]) -> Any:
        # optionally we should be able to pass in the index of the dataset we want
        # according to the order displayed when displayed as a pandas table
        if isinstance(key, int):
            key = self.all()[key]["id"]

        return self.get(dataset_id=key)

    def __delitem__(self, key: str) -> Any:
        self.delete(dataset_id=key)

    def to_obj(self, result: Any) -> Any:
        dataset_obj = super().to_obj(result)
        dataset_obj.pandas = DataFrame(dataset_obj.data)
        datasets = []

        pointers = self.node.store
        for data in dataset_obj.data:
            _class_name = ResponseObjectEnum.DATA.capitalize()
            data_obj = type(_class_name, (object,), data)()
            data_obj.shape = ast.literal_eval(data_obj.shape)
            data_obj.pointer = pointers[data_obj.id.replace("-", "")]
            datasets.append(data_obj)

        dataset_obj.files = datasets
        type(dataset_obj).__getitem__ = lambda x, i: x.data[i]
        dataset_obj.node = self.node
        return Dataset(dataset_obj)

    def _repr_html_(self) -> str:
        """Displays a nice table when the object is returned in Jupyter Notebook"""
        return self.all(pandas=True)._repr_html_()  # type: ignore


class Dataset:
    def __init__(self, dataset_metadata: Any) -> None:
        self.dataset_metadata = dataset_metadata
        self.id = self.dataset_metadata.id
        self.tags = self.dataset_metadata.tags
        self.manifest = self.dataset_metadata.manifest
        self.pandas = self.dataset_metadata.pandas

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            obj_id = self.dataset_metadata.data[key]["id"].replace("-", "")
            return self.dataset_metadata.node.store[obj_id]
        elif isinstance(key, str):
            return self.dataset_metadata.node.store[key.replace("-", "")]

    def _repr_html_(self) -> str:
        id = "<b>Id: </b>" + str(self.id) + "<br />"
        tags = "<b>Tags: </b>" + str(self.tags) + "<br />"
        manifest = "<b>Manifest: </b>" + str(self.manifest) + "<br /><br />"
        table = self.pandas._repr_html_()

        return self.__repr__() + "<br /><br />" + id + tags + manifest + table
