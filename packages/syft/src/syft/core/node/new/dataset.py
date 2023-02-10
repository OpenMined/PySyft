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
    name: str
    description: str
    contributors: List[Contributor]
    mock_is_real: bool = False


@serializable(recursive_serde=True)
class Dataset(SyftObject):
    # version
    __canonical_name__ = "Dataset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_uid: Optional[UID]
    assets: List[Asset]
    citation: Optional[str]
    url: Optional[str]
    description: Optional[str]

    __attr_searchable__ = ["name", "citation", "url", "description"]
    __attr_unique__ = ["name"]


class CreateDataset(Dataset):
    # version
    __canonical_name__ = "CreateDataset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None


def validate_url(obj: Any, state: Dict) -> Dict:
    state["url"] = GridURL.from_url(state["url"])
    return state


@transform(CreateDataset, Dataset)
def createdataset_to_dataset() -> List[Callable]:
    return [generate_id, validate_url]


class DatasetUpdate:
    pass
