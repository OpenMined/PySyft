# stdlib
import logging
import sys
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

# third party
import pandas as pd

# relative
from ..... import deserialize
from .....core.tensor.autodp.adp_tensor import ADPTensor
from .....core.tensor.tensor import Tensor
from ....common import UID
from ....common.serde.serialize import _serialize as serialize  # noqa: F401
from ...abstract.node import AbstractNodeClient
from ...enums import RequestAPIFields
from ...enums import ResponseObjectEnum
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
        response = self.node.conn.send_files(  # type: ignore
            "/datasets", path, form_name="metadata", form_values=kwargs
        )  # type: ignore
        logging.info(response[RequestAPIFields.MESSAGE])

    def all(self) -> List[Any]:
        result = [
            content
            for content in self.perform_api_request(
                syft_msg=self._get_all_message, timeout=1
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
        for key, _ in enumerate(a):
            raw = a[key : key + 1]  # noqa: E203
            out.append(Dataset(raw, self.client, key=key, **a[key]))
        return out

    def purge(self, skip_check: bool = False) -> None:
        if not skip_check:
            pref = input(
                "You are about to delete all datasets ? 🚨 \n"
                "All information will be permanantely deleted.\n"
                "Please enter y/n to proceed: "
            )
            while pref != "y" and pref != "n":
                pref = input("Invalid input '" + pref + "', please specify 'y' or 'n'.")
            if pref == "n":
                print("Datasets deletion is cancelled.")
                return None

        for dataset in self.all():
            self.delete(dataset_id=dataset.get("id"))

    def __len__(self) -> int:
        return len(self.all())

    def __delitem__(self, key: int) -> Any:

        try:
            dataset = self.all()[key]
        except IndexError as err:
            raise err

        dataset_id = dataset.get("id")
        dataset_name = dataset.get("name", "")

        pref = input(
            f"You are about to delete the `{dataset_name}` ? 🚨 \n"
            "All information related to this dataset will be permanantely deleted.\n"
            "Please enter y/n to proceed: "
        )
        while pref != "y" and pref != "n":
            pref = input("Invalid input '" + pref + "', please specify 'y' or 'n'.")
        if pref == "n":
            raise Exception("Dataset deletion is cancelled.")

        self.delete(dataset_id=dataset_id)
        sys.stdout.write(f"Dataset: `{dataset_name}` is successfully deleted.")

        return True

    def _repr_html_(self) -> str:
        if len(self) > 0:
            return self.dataset_list_to_html(self.all_as_datasets())
        return "(no datasets found)"

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

                <table id="myTable" style="width:1000px">
                  <tr class="header">
                    <th style="width:30px">Idx</th>
                    <th style="width:20%;">Name</th>
                    <th style="width:35%;">Description</th>
                    <th style="width:20%;">Assets</th>
                    <th style="width:300px;">Id</th>
                  </tr>
                """

        rows = ""
        for _, d in enumerate(dataset_iterable):

            data = d.data
            truncated_assets = False
            if len(data) > 3:
                truncated_assets = True
                data = data[:3]

            assets = ""
            for _, a in enumerate(data):
                assets += '["' + a["name"] + '"] -> ' + a["dtype"] + "<br /><br />"

            if truncated_assets:
                assets += "...<br /><br />"

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
        tags: Optional[List[str]] = None,
    ) -> None:
        self.raw = raw
        self.description = description
        self.name = name
        self.id = id
        self.tags = tags if tags is not None else []
        self.data = data
        self.client = client
        self.key = key

    @property
    def pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.raw)

    @property
    def assets(self) -> Any:
        return self.data

    def __getitem__(self, key: str) -> Any:
        keys = list()
        for d in self.data:
            if d["name"] == key:
                return self.client.store.get(d["id"])  # type: ignore
            keys.append(d["name"])

        raise KeyError(
            "Asset '" + key + "' doesn't exist! Try one of these: " + str(keys)
        )

    def _repr_html_(self) -> str:

        print("Dataset: " + self.name)
        print("Description: " + self.description)
        print()

        rows = ""

        data = self.data

        if len(data) > 15:
            print(
                "WARNING: Too many assets to print... truncating... You "
                "may run \n\n assets = my_dataset.assets \n\nto view receive a "
                "dictionary you can parse through using Python\n(as opposed to blowing up your notebook"
                " with a massive printed table).\n"
            )
            data = data[0:15]

        assets = ""
        for _, a in enumerate(data):
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

    def add(self, name: str, value: Any, skip_checks: bool = False) -> None:
        """Add a new asset to the dataset.

        Args:
            name (str): Name of the asset
            value (dict): Value of the asset
        """

        # relative
        from .....lib.python.util import downcast

        if not skip_checks:
            if not isinstance(value, Tensor) or not isinstance(
                getattr(value, "child", None), ADPTensor
            ):
                raise Exception(
                    "ERROR: all private assets must be NumPy ndarray.int32 assets "
                    + "with proper Differential Privacy metadata applied.\n"
                    + "\n"
                    + "Example: syft.Tensor(np.ndarray([1,2,3,4]).astype(np.int32)).private()\n\n"
                    + "and then follow the wizard. 🧙"
                )
                # print(
                #     "\n\nWARNING - Non-DP Asset: You just passed in a asset '"
                #     + name
                #     + "' which cannot be tracked with differential privacy because it is a "
                #     + str(type(value))
                #     + " object.\n\n"
                #     + "This means you'll need to manually approve any requests which "
                #     + "leverage this data. If this is ok with you, proceed. If you'd like to use "
                #     + "automatic differential privacy budgeting, please pass in a DP-compatible tensor type "
                #     + "such as by calling .private() on a sy.Tensor with a np.int32 or np.float32 inside."
                # )
                #
                # pref = input("Are you sure you want to proceed? (y/n)")
                #
                # while pref != "y" and pref != "n":
                #     pref = input(
                #         "Invalid input '" + pref + "', please specify 'y' or 'n'."
                #     )
                # if pref == "n":
                #     raise Exception("Dataset loading cancelled.")

            existing_asset_names = [d.get("name") for d in self.data]
            if name in existing_asset_names:
                raise KeyError(
                    f"Asset with name: `{name}` already exists. "
                    "Please use a different name."
                )

            asset = {name: value}
            asset = downcast(asset)
            binary_dataset = serialize(asset, to_bytes=True)

            metadata = {"dataset_id": bytes(str(self.id), "utf-8")}
            metadata = downcast(metadata)

            sys.stdout.write("\rLoading dataset... uploading...")
            # Add a new asset to the dataset pointer
            DatasetRequestAPI(self.client).create_syft(
                dataset=binary_dataset, metadata=metadata, platform="syft"
            )
            sys.stdout.write("\rLoading dataset... uploading... \nSUCCESS!")
            self.refresh()

    def delete(self, name: str, skip_check: bool = False) -> bool:
        """Delete the asset with the given name."""

        asset_id = None

        for d in self.data:
            if d["name"] == name:
                asset_id = d["id"]  # Id of the first matching name
                break

        if asset_id is None:
            raise KeyError(f"The asset with name `{name}` does not exists.")

        if not skip_check:
            pref = input(
                f"You are about to permanantely delete the asset `{name}` ? 🚨 \n"
                "Please enter y/n to proceed: "
            )
            while pref != "y" and pref != "n":
                pref = input("Invalid input '" + pref + "', please specify 'y' or 'n'.")
            if pref == "n":
                sys.stdout.write("Asset deletion cancelled.")
                return False

        DatasetRequestAPI(self.client).delete(
            dataset_id=self.id, bin_object_id=asset_id
        )
        self.refresh()

        return True

    def refresh(self) -> None:
        """Update data to its latest state."""

        datasets = DatasetRequestAPI(self.client).all()
        self.data = datasets[self.key].get("data", [])

    def iter(self, exclude: Optional[List[str]] = None) -> Iterable:
        """Generate an asset iterable."""

        exclude = [] if exclude is None else exclude

        for asset in self.data:
            asset_name = asset["name"]
            if asset_name not in exclude:
                asset_id = asset["id"].replace("-", "")
                yield self.client.store.get(asset_id)  # type: ignore
