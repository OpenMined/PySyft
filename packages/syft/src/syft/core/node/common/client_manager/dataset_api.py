# stdlib
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
import pandas as pd

# relative
from ..... import deserialize
from ....common import UID
from ...abstract.node import AbstractNodeClient
from ...domain.enums import RequestAPIFields
from ...domain.enums import ResponseObjectEnum
from ..node_service.dataset_manager.dataset_manager_messages import CreateDatasetMessage
from ..node_service.dataset_manager.dataset_manager_messages import DeleteDatasetMessage
from ..node_service.dataset_manager.dataset_manager_messages import GetDatasetMessage
from ..node_service.dataset_manager.dataset_manager_messages import GetDatasetsMessage
from ..node_service.dataset_manager.dataset_manager_messages import UpdateDatasetMessage
from .request_api import RequestAPI

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
          width: 50%; /* Full-width */
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

        <table id="myTable">
          <tr class="header">
            <th style="width:15%;">Asset Key</th>
            <th style="width:20%;">Type</th>
            <th style="width:10%;">Shape</th>
          </tr>
        """

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
            asset_td = tr[i].getElementsByTagName("td")[3];
            id_td = tr[i].getElementsByTagName("td")[4];
            if (name_td || desc_td || asset_td || id_td) {
              name_txtValue = name_td.textContent || name_td.innerText;
              desc_txtValue = desc_td.textContent || name_td.innerText;
              asset_txtValue = asset_td.textContent || name_td.innerText;
              id_txtValue = id_td.textContent || name_td.innerText;
              name_bool = name_txtValue.toUpperCase().indexOf(filter) > -1;
              desc_bool = desc_txtValue.toUpperCase().indexOf(filter) > -1;
              asset_bool = asset_txtValue.toUpperCase().indexOf(filter) > -1;
              id_bool = id_txtValue.toUpperCase().indexOf(filter) > -1;
              if (name_bool || desc_bool || asset_bool || id_bool) {
                tr[i].style.display = "";
              } else {
                tr[i].style.display = "none";
              }
            }
          }
        }
        </script>"""


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
            d = a[key : key + 1]  # noqa: E203
            return Dataset(d, self.client, key=key, **a[key])

        elif isinstance(key, slice):

            class NewObject:
                def _repr_html_(self2: Any) -> str:
                    return self.dataset_list_to_html(
                        self.all_as_datasets().__getitem__(key)  # type: ignore
                    )

            return NewObject()

    def all_as_datasets(self) -> List[Any]:
        a = self.all()
        out = list()
        for key, d in enumerate(a):
            raw = a[key : key + 1]  # noqa: E203
            out.append(Dataset(raw, self.client, key=key, **a[key]))
        return out

    def __len__(self) -> int:
        return len(self.all())

    def __delitem__(self, key: str) -> Any:
        self.delete(dataset_id=key)

    def _repr_html_(self) -> str:
        if len(self) > 0:
            return self.dataset_list_to_html(self.all_as_datasets())
        return "(no datasets found)Z"

    @staticmethod
    def dataset_list_to_html(dataset_iterable: List[Any]) -> str:

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
                    <th style="width:35%;">Description</th>
                    <th style="width:20%;">Assets</th>
                    <th style="width:300px;">Id</th>
                  </tr>
                """

        rows = ""
        for row_i, d in enumerate(dataset_iterable):
            assets = ""
            for i, a in enumerate(d.data):
                assets += '["' + a["name"] + '"] -> ' + a["dtype"] + "<br /><br />"

            rows += (
                """

          <tr>
            <td>["""
                + str(d.key)
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
                + d.id
                + """</td>
          </tr>"""
            )

        return initial_boilerplate + rows + end_boilerplate


class Dataset:
    def __init__(
        self,
        raw: Any,
        client: AbstractNodeClient,
        description: str,
        name: str,
        id: UID,
        key: int,
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
        self.key = key

    @property
    def pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.raw)

    def __getitem__(self, key: str) -> Any:
        keys = list()
        for d in self.data:
            if d["name"] == key:
                return self.client.store[d["id"].replace("-", "")]  # type: ignore
            keys.append(d["name"])

        raise KeyError(
            "Asset '" + key + "' doesn't exist! Try one of these: " + str(keys)
        )

    def _repr_html_(self) -> str:

        print("Dataset: " + self.name)
        print("Description: " + self.description)
        print()

        rows = ""

        assets = ""
        for i, a in enumerate(self.data):
            assets += '["' + a["name"] + '"] -> ' + a["dtype"] + "<br /><br />"

            rows += (
                """

              <tr>
            <td>[\""""
                + a["name"]
                + """\"]</td>
            <td>"""
                + a["dtype"]
                + """</td>
            <td>"""
                + a["shape"]
                + """</td>
          </tr>"""
            )
        end_boilerplate = """
        </table>

        """

        return initial_boilerplate + rows + end_boilerplate
