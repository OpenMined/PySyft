# stdlib
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID

# from .action_object import ActionObjectPointer
from .data_subject import DataSubject
from .document_store import PartitionKey
from .response import SyftError
from .response import SyftSuccess
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
    twin_uid: UID
    mock_is_real: bool = False
    shape: Tuple

    # @property
    # def pointer(self) -> ActionObjectPointer:
    #     api = APIRegistry.api_for(node_uid=self.node_uid)
    #     obj_ptr = api.services.action.get_pointer(uid=self.action_id)
    #     return obj_ptr

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

    @property
    def mock(self) -> Any:
        # relative
        from .api import APIRegistry

        api = APIRegistry.api_for(node_uid=self.node_uid)
        return api.services.action.get_pointer(self.twin_uid)


@serializable(recursive_serde=True)
class CreateAsset(SyftObject):
    # version
    __canonical_name__ = "CreateAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None
    name: str
    description: Optional[str]
    contributors: List[Contributor] = []
    data_subjects: List[DataSubject] = []
    action_id: Optional[UID]
    node_uid: Optional[UID]
    data: Optional[Any]
    mock: Optional[Any]
    shape: Optional[Tuple]
    mock_is_real: bool = False

    def add_data_subject(self, data_subject: DataSubject) -> None:
        self.data_subjects.append(data_subject)

    def add_contributor(
        self,
        name: str,
        email: str,
        role: Union[Enum, str],
        phone: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        _role_str = role.value if isinstance(role, Enum) else role
        contributor = Contributor(
            name=name, role=_role_str, email=email, phone=phone, note=note
        )
        self.contributors.append(contributor)

    def set_description(self, description: str) -> None:
        self.description = description

    def set_obj(self, data: Any) -> None:
        self.data = data

    def set_mock(self, mock_data: Any, mock_is_real: bool) -> None:
        self.mock = mock_data
        self.mock_is_real = mock_is_real

    def set_shape(self, shape: Tuple) -> None:
        self.shape = shape

    def check(self) -> Union[SyftSuccess, SyftError]:
        if type(self.data) != type(self.mock):  # noqa: E721
            return SyftError(
                message=f"set_obj type {type(self.data)} must match set_mock type {type(self.mock)}"
            )
        data_shape = get_shape_or_len(self.data)
        mock_shape = get_shape_or_len(self.mock)
        if data_shape != mock_shape:
            return SyftError(
                message=f"set_obj shape {data_shape} must match set_mock shape {mock_shape}"
            )

        return SyftSuccess(message="Dataset is Valid")


def get_shape_or_len(obj: Any) -> Optional[Union[Tuple[int, ...], int]]:
    shape = getattr(obj, "shape", None)
    if shape:
        return shape
    len_attr = getattr(obj, "__len__", None)
    if len_attr is not None:
        return len_attr()
    return None


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
        role: Union[Enum, str],
        phone: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        _role_str = role.value if isinstance(role, Enum) else role
        contributor = Contributor(
            name=name, role=_role_str, email=email, phone=phone, note=note
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

    def check(self) -> Result[SyftSuccess, List[SyftError]]:
        errors = []
        for asset in self.asset_list:
            result = asset.check()
            if not result:
                errors.append(result)
        if len(errors):
            return Err(errors)
        return Ok(SyftSuccess(message="Dataset is Valid"))


def create_and_store_twin(context: TransformContext) -> TransformContext:
    # relative
    from .twin_object import TwinObject

    twin = TwinObject(
        private_obj=context.output.pop("data", None),
        mock_obj=context.output.pop("mock", None),
    )
    action_service = context.node.get_service("actionservice")
    result = action_service.set(context=context.to_node_context(), action_object=twin)
    if result.is_err():
        raise Exception(f"Failed to create and store twin. {result}")

    context.output["twin_uid"] = twin.id
    return context


def infer_shape(context: TransformContext) -> TransformContext:
    if context.output["shape"] is None:
        context.output["shape"] = get_shape_or_len(context.obj.mock)
    return context


@transform(CreateAsset, Asset)
def createasset_to_asset() -> List[Callable]:
    return [generate_id, infer_shape, create_and_store_twin]


def convert_asset(context: TransformContext) -> TransformContext:
    assets = context.output.pop("asset_list", [])
    for idx, asset in enumerate(assets):
        assets[idx] = asset.to(Asset, context=context)
    context.output["asset_list"] = assets
    return context


@transform(CreateDataset, Dataset)
def createdataset_to_dataset() -> List[Callable]:
    return [generate_id, validate_url, convert_asset]


class DatasetUpdate:
    pass
