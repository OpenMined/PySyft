# stdlib
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
import pandas as pd

# syft absolute
from syft import deserialize
from syft.core.common import UID
from syft.core.node.abstract.node import AbstractNodeClient
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    CreateDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    DeleteDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetsMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    UpdateDatasetMessage,
)

# relative
from ....node.domain.enums import RequestAPIFields
from ....node.domain.enums import ResponseObjectEnum
from ...common.client_manager.request_api import RequestAPI


class DatasetRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            create_msg=CreateDatasetMessage,
            get_msg=GetDatasetMessage,
            get_all_msg=GetDatasetsMessage,
            update_msg=UpdateDatasetMessage,
            delete_msg=DeleteDatasetMessage,
            response_key=ResponseObjectEnum.DATASET,
        )

    def create_syft(self, **kwargs: Any) -> None:
        super().create(**kwargs)

    def create_grid_ui(self, path: str, **kwargs) -> Dict[str, str]:  # type: ignore
        response = self.node.conn.send_files(path, metadata=kwargs)  # type: ignore
        logging.info(response[RequestAPIFields.MESSAGE])

    def all(self) -> List[Any]:
        result = [
            content
            for content in self.perform_api_request(
                syft_msg=self._get_all_message
            ).metadatas
        ]

        new_all = list()
        for dataset in result:
            new_dataset = {}
            for k, v_blob in dataset.items():
                if k not in ["str_metadata", "blob_metadata", "manifest"]:
                    new_dataset[k] = deserialize(v_blob, from_bytes=True)
            new_all.append(new_dataset)

        return new_all

    def __getitem__(self, key: Union[str, int, slice]) -> Any:
        # optionally we should be able to pass in the index of the dataset we want
        # according to the order displayed when displayed as a pandas table

        if isinstance(key, int):
            a = self.all()
            return Dataset(a[key : key + 1], self.client, **a[key])  # noqa: E203

    def all_as_datasets(self):
        a = self.all()
        out = list()
        for key, d in enumerate(a):
            out.append(Dataset(a[key : key + 1], self.client, **a[key]))  # noqa: E203
        return out

    def __len__(self):
        return len(self.all())

    def __delitem__(self, key: str) -> Any:
        self.delete(dataset_id=key)

    def _repr_html_(self):
        initial_boilerplate = """<style>
        #myInput {
          background-position: 10px 12px; /* Position the search icon */
          background-repeat: no-repeat; /* Do not repeat the icon image */
          background-color: #bbb;
          width: 98%; /* Full-width */
          font-size: 14px; /* Increase font-size */
          padding: 12px 20px 12px 40px; /* Add some padding */
          border: 1px solid #ddd; /* Add a grey border */
          margin-bottom: 12px; /* Add some space below the input */
        }

        #myTable {
          border-collapse: collapse; /* Collapse borders */
          width: 100%; /* Full-width */
          border: 1px solid #ddd; /* Add a grey border */
          font-size: 14px; /* Increase font-size */
        }

        #myTable th, #myTable td {
          text-align: left; /* Left-align text */
          padding: 10px; /* Add padding */
        }

        #myTable tr {
          /* Add a bottom border to all table rows */
          border-bottom: 1px solid #ddd;
        }

        #myTable tr.header, #myTable tr:hover {
          /* Add a grey background color to the table header and on hover */
          background-color: #777;
        }
        </style>

        <input type="text" id="myInput" onkeyup="myFunction()" placeholder="Search for datasets..">

        <table id="myTable">
          <tr class="header">
            <th style="width:30px">Idx</th>
            <th style="width:20%;">Name</th>
            <th style="width:45%;">Description</th>
            <th style="width:15%;">Assets</th>
            <th style="width:300px;">Id</th>
          </tr>
        """

        rows = ""
        for row_i, d in enumerate(self.all_as_datasets()):
            assets = ""
            for i, a in enumerate(d.data):
                assets += '["' + a["name"] + '"] -> ' + a["dtype"] + "<br /><br />"

            rows += (
                """

          <tr>
            <td>["""
                + str(row_i)
                + """]</td>
            <td>"""
                + d.name
                + """</td>
            <td>"""
                + d.description[0:500]
                + """</td>
            <td>"""
                + assets
                + """</td>
            <td>"""
                + d.id.replace("-", "")
                + """</td>
          </tr>"""
            )
        end_boilerplate = """
        </table>

        <script>
        function myFunction() {
          // Declare variables
          var input, filter, table, tr, td, i, txtValue;
          input = document.getElementById("myInput");
          filter = input.value.toUpperCase();
          table = document.getElementById("myTable");
          tr = table.getElementsByTagName("tr");

          // Loop through all table rows, and hide those who don't match the search query
          for (i = 0; i < tr.length; i++) {
            name_td = tr[i].getElementsByTagName("td")[1];
            desc_td = tr[i].getElementsByTagName("td")[2];
            if (name_td || desc_td) {
              name_txtValue = name_td.textContent || name_td.innerText;
              desc_txtValue = desc_td.textContent || name_td.innerText;
              if (name_txtValue.toUpperCase().indexOf(filter) > -1 || desc_txtValue.toUpperCase().indexOf(filter) > -1) {
                tr[i].style.display = "";
              } else {
                tr[i].style.display = "none";
              }
            }
          }
        }
        </script>"""

        return initial_boilerplate + rows + end_boilerplate

    # def to_obj(self, result: Any) -> Any:
    #     dataset_obj = super().to_obj(result)
    #     dataset_obj.pandas = DataFrame(dataset_obj.data)
    #     datasets = []
    #
    #     pointers = self.client.store
    #     for data in dataset_obj.data:
    #         _class_name = ResponseObjectEnum.DATA.capitalize()
    #         data_obj = type(_class_name, (object,), data)()
    #         data_obj.shape = ast.literal_eval(data_obj.shape)
    #         data_obj.pointer = pointers[data_obj.id.replace("-", "")]
    #         datasets.append(data_obj)
    #
    #     dataset_obj.files = datasets
    #     type(dataset_obj).__getitem__ = lambda x, i: x.data[i]
    #     dataset_obj.node = self.client
    #     return Dataset(dataset_obj)


class Dataset:
    def __init__(
        self,
        raw: Any,
        client: AbstractNodeClient,
        description: str,
        name: str,
        id: UID,
        data: Any,
        tags: List[str] = [],
    ) -> None:
        self.raw = raw
        self.description = description
        self.name = name
        self.id = id
        self.tags = tags
        self.data = data
        self.client = client

    @property
    def pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.raw)

    def __getitem__(self, key: str) -> Any:
        for d in self.data:
            if d["name"] == key:
                return self.client.store[d["id"].replace("-", "")]

    def _repr_html_(self) -> str:
        return self.pandas._repr_html_()


# class Dataset:
#     def __init__(self, dataset_metadata: Any) -> None:
#         self.dataset_metadata = dataset_metadata
#         self.id = self.dataset_metadata.id
#         self.description = self.dataset_metadata.description
#         self.tags = self.dataset_metadata.tags
#         self.manifest = self.dataset_metadata.manifest
#         self.pandas = self.dataset_metadata.pandas
#
#         for key, value in self.dataset_metadata.str_metadata.items():
#             setattr(self, key, value)
#
#         for key, value in self.dataset_metadata.blob_metadata.items():
#             setattr(self, key, deserialize(b"".fromhex(value), from_bytes=True))
#
#     def __getitem__(self, key: Union[str, int]) -> Any:
#         if isinstance(key, int):
#             obj_id = self.dataset_metadata.data[key]["id"].replace("-", "")
#             return self.dataset_metadata.node.store[obj_id]
#         elif isinstance(key, str):
#             id = self.dataset_metadata.pandas[
#                 self.dataset_metadata.pandas["name"] == key
#             ].id.values[0]
#             return self.dataset_metadata.node.store[id.replace("-", "")]
#
#     def _repr_html_(self) -> str:
#         id = "<b>Id: </b>" + str(self.id) + "<br />"
#         tags = "<b>Tags: </b>" + str(self.tags) + "<br />"
#         manifest = "<b>Manifest: </b>" + str(self.manifest) + "<br /><br />"
#         table = self.pandas._repr_html_()
#
#         return self.__repr__() + "<br /><br />" + id + tags + manifest + table
