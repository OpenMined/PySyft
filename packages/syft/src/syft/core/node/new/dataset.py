# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import numpy as np

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_object import ActionObjectPointer
from .api import APIRegistry
from .data_subject import DataSubject
from .document_store import PartitionKey
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform
from .transforms import validate_url

NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable(recursive_serde=True)
class Contributor(SyftObject):
    __canonical_name__ = "Contributor"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    role: str
    email: str
    phone: Optional[str]
    note: Optional[str]

    __attr_repr_cols__ = ["name", "role", "email"]


@serializable(recursive_serde=True)
class Asset(SyftObject):
    # version
    __canonical_name__ = "Asset"
    __version__ = SYFT_OBJECT_VERSION_1

    action_id: UID
    node_uid: UID
    name: str
    description: Optional[str]
    contributors: List[Contributor] = []
    data_subjects: List[DataSubject] = []
    mock: Any
    mock_is_real: bool = False
    shape: Tuple

    @property
    def pointer(self) -> ActionObjectPointer:
        api = APIRegistry.api_for(node_uid=self.node_uid)
        obj_ptr = api.services.action.get_pointer(uid=self.action_id)
        return obj_ptr

    def _repr_markdown_(self) -> str:
        _repr_str = f"Asset: {self.name}\n"
        _repr_str += f"Pointer Id: {self.action_id}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Total Data Subjects: {len(self.data_subjects)}\n"
        _repr_str += f"Shape: {self.shape}\n"
        _repr_str += f"Contributors: {len(self.contributors)}\n"
        for contributor in self.contributors:
            _repr_str += f"\t{contributor.name}: {contributor.email}\n"
        return "```python\n" + _repr_str + "\n```"


@serializable(recursive_serde=True)
class CreateAsset(Asset):
    # version
    __canonical_name__ = "CreateAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None
    action_id: Optional[UID]
    node_uid: Optional[UID]
    data: Optional[np.ndarray]
    mock: Optional[Any]
    shape: Optional[Tuple]

    def add_data_subject(self, data_subject: DataSubject) -> None:
        self.data_subjects.append(data_subject)

    def add_contributor(
        self,
        name: str,
        email: str,
        role: str,
        phone: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        contributor = Contributor(
            name=name, role=role, email=email, phone=phone, note=note
        )
        self.contributors.append(contributor)

    def set_description(self, description: str) -> None:
        self.description = description

    def set_obj(self, data: np.ndarray) -> None:
        self.data = data

    def set_mock(self, mock_data: Any, mock_is_real: bool) -> None:
        self.mock = mock_data
        self.mock_is_real = mock_is_real

    def set_shape(self, shape: Tuple) -> None:
        self.shape = shape


@serializable(recursive_serde=True)
class Dataset(SyftObject):
    # version
    __canonical_name__ = "Dataset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_uid: Optional[UID]
    asset_list: List[Asset] = []
    contributors: List[Contributor] = []
    citation: Optional[str]
    url: Optional[str]
    description: Optional[str]

    __attr_searchable__ = ["name", "citation", "url", "description"]
    __attr_unique__ = ["name"]
    __attr_repr_cols__ = ["name", "url"]

    @property
    def assets(self) -> Dict[str, str]:
        data = {}
        for asset in self.asset_list:
            data[asset.name] = asset
        return data

    def _repr_markdown_(self) -> str:
        _repr_str = f"Syft Dataset: {self.name}\n"
        _repr_str += "Assets:\n"
        for asset in self.asset_list:
            _repr_str += f"\t{asset.name}: {asset.description}\n"
        if self.citation:
            _repr_str += f"Citation: {self.citation}\n"
        if self.url:
            _repr_str += f"URL: {self.url}\n"
        if self.description:
            _repr_str += f"Description: {self.description}\n"
        return "```python\n" + _repr_str + "\n```"


@serializable(recursive_serde=True)
class CreateDataset(Dataset):
    # version
    __canonical_name__ = "CreateDataset"
    __version__ = SYFT_OBJECT_VERSION_1
    asset_list: List[CreateAsset] = []

    id: Optional[UID] = None

    def set_description(self, description: str) -> None:
        self.description = description

    def add_citation(self, citation: str) -> None:
        self.citation = citation

    def add_url(self, url: str) -> None:
        self.url = url

    def add_contributor(
        self,
        name: str,
        email: str,
        role: str,
        phone: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        contributor = Contributor(
            name=name, role=role, email=email, phone=phone, note=note
        )
        self.contributors.append(contributor)

    def add_asset(self, asset: CreateAsset) -> None:
        self.asset_list.append(asset)

    def remove_asset(self, name: str) -> None:
        asset_to_remove = None
        for asset in self.asset_list:
            if asset.name == name:
                asset_to_remove = asset
                break

        if asset_to_remove is None:
            print(f"No asset exists with name: {name}")
        self.asset_list.remove(asset_to_remove)


def remove_original_data(context: TransformContext) -> TransformContext:
    context.output.pop("data", None)
    return context


@transform(CreateAsset, Asset)
def createasset_to_asset() -> List[Callable]:
    return [generate_id, remove_original_data]


def convert_asset(context: TransformContext) -> TransformContext:
    assets = context.output.pop("asset_list", [])
    for idx, asset in enumerate(assets):
        assets[idx] = asset.to(Asset)
    context.output["asset_list"] = assets
    return context


@transform(CreateDataset, Dataset)
def createdataset_to_dataset() -> List[Callable]:
    return [generate_id, validate_url, convert_asset]


class DatasetUpdate:
    pass
