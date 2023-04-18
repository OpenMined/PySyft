# stdlib
from collections import OrderedDict
from datetime import datetime
from enum import Enum
import sys
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.transforms import validate_url
from ...types.uid import UID
from ..data_subject.data_subject import DataSubject
from ..data_subject.data_subject import DataSubjectCreate
from ..data_subject.data_subject_service import DataSubjectService
from ..response import SyftError
from ..response import SyftException
from ..response import SyftSuccess


@serializable()
class TupleDict(OrderedDict):
    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super(TupleDict, self).__getitem__(key)


NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable()
class Contributor(SyftObject):
    __canonical_name__ = "Contributor"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    role: str
    email: str
    phone: Optional[str]
    note: Optional[str]

    __attr_repr_cols__ = ["name", "role", "email"]


@serializable()
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
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(node_uid=self.node_uid)
        return api.services.action.get_pointer(self.action_id)

    @property
    def data(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(node_uid=self.node_uid)
        return api.services.action.get(self.action_id)


@serializable()
class CreateAsset(SyftObject):
    # version
    __canonical_name__ = "CreateAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None
    name: str
    description: Optional[str]
    contributors: List[Contributor] = []
    data_subjects: List[DataSubjectCreate] = []
    node_uid: Optional[UID]
    action_id: Optional[UID]
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
        if isinstance(data, SyftError):
            raise SyftException(data)
        self.data = data

    def set_mock(self, mock_data: Any, mock_is_real: bool) -> Any:
        if isinstance(mock_data, SyftError):
            raise SyftException(mock_data)
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
    if hasattr(obj, "shape"):
        shape = getattr(obj, "shape", None)
        if shape:
            return shape
    len_attr = getattr(obj, "__len__", None)
    if len_attr is not None:
        return len_attr()
    return None


@serializable()
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
    updated_at: Optional[str]
    requests: Optional[int] = 0
    mb_size: Optional[int]

    __attr_searchable__ = ["name", "citation", "url", "description", "action_ids"]
    __attr_unique__ = ["name"]
    __attr_repr_cols__ = ["name", "url"]

    def action_ids(self) -> List[UID]:
        data = []
        for asset in self.asset_list:
            if asset.action_id:
                data.append(asset.action_id)
        return data

    @property
    def assets(self) -> TupleDict:
        data = TupleDict()
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

    @property
    def client(self) -> Optional[Any]:
        # relative
        from ...client.client import SyftClientSessionCache

        client = SyftClientSessionCache.get_client_for_node_uid(self.node_uid)
        if client is None:
            return SyftError(
                message=f"No clients for {self.node_uid} in memory. Please login with sy.login"
            )
        return client


@serializable()
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
    action_id = context.output["action_id"]
    if action_id is None:
        # relative
        from ...types.twin_object import TwinObject

        private_obj = context.output.pop("data", None)
        mock_obj = context.output.pop("mock", None)
        if private_obj is None and mock_obj is None:
            raise Exception("No data and no action_id means this asset has no data")

        twin = TwinObject(
            private_obj=private_obj,
            mock_obj=mock_obj,
        )
        action_service = context.node.get_service("actionservice")
        result = action_service.set(
            context=context.to_node_context(), action_object=twin
        )
        if result.is_err():
            raise Exception(f"Failed to create and store twin. {result}")

        context.output["action_id"] = twin.id
    else:
        private_obj = context.output.pop("data", None)
        mock_obj = context.output.pop("mock", None)
    return context


def infer_shape(context: TransformContext) -> TransformContext:
    if context.output["shape"] is None:
        context.output["shape"] = get_shape_or_len(context.obj.mock)
    return context


def set_data_subjects(context: TransformContext) -> TransformContext:
    data_subjects = context.output["data_subjects"]
    get_data_subject = context.node.get_service_method(DataSubjectService.get_by_name)

    resultant_data_subjects = []
    for data_subject in data_subjects:
        result = get_data_subject(context=context, name=data_subject.name)
        if isinstance(result, SyftError):
            return result
        resultant_data_subjects.append(result)
    context.output["data_subjects"] = resultant_data_subjects
    return context


@transform(CreateAsset, Asset)
def createasset_to_asset() -> List[Callable]:
    return [generate_id, infer_shape, create_and_store_twin, set_data_subjects]


def convert_asset(context: TransformContext) -> TransformContext:
    assets = context.output.pop("asset_list", [])
    dataset_size = 0
    for idx, create_asset in enumerate(assets):
        dataset_size += sys.getsizeof(assets) / 1024
        asset_context = TransformContext.from_context(obj=create_asset, context=context)
        assets[idx] = create_asset.to(Asset, context=asset_context)
    context.output["asset_list"] = assets
    context.output["mb_size"] = dataset_size
    return context


def add_current_date(context: TransformContext) -> TransformContext:
    current_date = datetime.now()
    formatted_date = current_date.strftime("%b %d, %Y")
    context.output["updated_at"] = formatted_date
    return context


@transform(CreateDataset, Dataset)
def createdataset_to_dataset() -> List[Callable]:
    return [generate_id, validate_url, convert_asset, add_current_date]


class DatasetUpdate:
    pass
