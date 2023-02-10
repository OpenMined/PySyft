# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ....core.node.common.node_table.syft_object import transform
from ....grid import GridURL
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_object import ActionObjectPointer
from .api import APIRegistry
from .document_store import CollectionKey
from .transforms import generate_id

NameCollectionKey = CollectionKey(key="name", type_=str)


@serializable(recursive_serde=True)
class Contributor(SyftObject):
    name: str
    role: str
    email: str
    phone: Optional[str]
    note: Optional[str]


@serializable(recursive_serde=True)
class Asset(SyftObject):
    # version
    __canonical_name__ = "Asset"
    __version__ = SYFT_OBJECT_VERSION_1

    action_id: UID
    node_uid: UID
    name: str
    description: str
    contributors: List[Contributor]
    mock_is_real: bool = False

    @property
    def pointer(self) -> ActionObjectPointer:
        api = APIRegistry.api_for(node_uid=self.node_uid)
        obj_ptr = api.services.action.get_pointer(uid=self.action_id)
        return obj_ptr


@serializable(recursive_serde=True)
class Dataset(SyftObject):
    # version
    __canonical_name__ = "Dataset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_uid: Optional[UID]
    asset_list: List[Asset]
    citation: Optional[str]
    url: Optional[str]
    description: Optional[str]

    __attr_searchable__ = ["name", "citation", "url", "description"]
    __attr_unique__ = ["name"]

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

    id: Optional[UID] = None


def validate_url(obj: Any, state: Dict) -> Dict:
    if state["url"] is not None:
        state["url"] = GridURL.from_url(state["url"]).url_no_port
    return state


@transform(CreateDataset, Dataset)
def createdataset_to_dataset() -> List[Callable]:
    return [generate_id, validate_url]


class DatasetUpdate:
    pass
