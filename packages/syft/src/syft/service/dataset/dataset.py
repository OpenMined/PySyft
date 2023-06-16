# stdlib
from collections import OrderedDict
from datetime import datetime
from enum import Enum
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import itables
from pydantic import ValidationError
from pydantic import root_validator
from pydantic import validator
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.transforms import validate_url
from ...types.uid import UID
from ...util import options
from ...util.colors import ON_SURFACE_HIGHEST
from ...util.colors import SURFACE
from ...util.colors import SURFACE_SURFACE
from ...util.fonts import ITABLES_CSS
from ...util.fonts import fonts_css
from ...util.markdown import as_markdown_python_code
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
    role: Optional[str]
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
    shape: Optional[Tuple]
    created_at: DateTime = DateTime.now()

    __attr_repr_cols__ = ["name", "shape"]

    def _repr_html_(self) -> Any:
        itables_css = f"""
        .itables table {{
            margin: 0 auto;
            float: left;
            color: {ON_SURFACE_HIGHEST[options.color_theme]};
        }}
        .itables table th {{color: {SURFACE_SURFACE[options.color_theme]};}}
        """

        # relative
        from ...service.action.action_object import ActionObject

        uploaded_by_line = ""
        if len(self.contributors) > 0:
            uploaded_by_line = (
                f"<p><strong>Uploaded by: </strong>{self.contributors[0].name}</p>"
            )
        if isinstance(self.data, ActionObject):
            data_table_line = itables.to_html_datatable(
                df=self.data.syft_action_data, css=itables_css
            )
        else:
            data_table_line = self.data
        return (
            f"""
            <style>
            {fonts_css}
            .syft-asset {{color: {SURFACE[options.color_theme]};}}
            .syft-asset h3,
            .syft-asset p
              {{font-family: 'Open Sans'}}
            {ITABLES_CSS}
            </style>
            """
            + '<div class="syft-asset">'
            + f"<h3>{self.name}</h3>"
            + f"<p>{self.description}</p>"
            + f"<p><strong>Asset ID: </strong>{self.id}</p>"
            + f"<p><strong>Action Object ID: </strong>{self.action_id}</p>"
            + uploaded_by_line
            + f"<p><strong>Created on: </strong>{self.created_at}</p>"
            + "<p><strong>Data:</strong></p>"
            + data_table_line
            + "<p><strong>Mock Data:</strong></p>"
            + itables.to_html_datatable(df=self.mock, css=itables_css)
            + "</div>"
        )

    def _repr_markdown_(self) -> str:
        _repr_str = f"Asset: {self.name}\n"
        _repr_str += f"Pointer Id: {self.action_id}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Total Data Subjects: {len(self.data_subjects)}\n"
        _repr_str += f"Shape: {self.shape}\n"
        _repr_str += f"Contributors: {len(self.contributors)}\n"
        for contributor in self.contributors:
            _repr_str += f"\t{contributor.name}: {contributor.email}\n"
        return as_markdown_python_code(_repr_str)

    @property
    def pointer(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.action.get_pointer(self.action_id)

    @property
    def mock(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.action.get_pointer(self.action_id).syft_action_data

    @property
    def data(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.action.get(self.action_id)


def _is_action_data_empty(obj: Any) -> bool:
    # just a wrapper of action_object.is_action_data_empty
    # to work around circular import error

    # relative
    from ...service.action.action_object import is_action_data_empty

    return is_action_data_empty(obj)


def check_mock(data: Any, mock: Any) -> bool:
    if type(data) == type(mock):
        return True

    return _is_action_data_empty(mock)


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
    created_at: Optional[DateTime]

    __attr_repr_cols__ = ["name"]

    class Config:
        validate_assignment = True

    @root_validator()
    def __empty_mock_cannot_be_real(cls, values: dict[str, Any]) -> Dict:
        """set mock_is_real to False whenever mock is None or empty"""

        if (mock := values.get("mock")) is None or _is_action_data_empty(mock):
            values["mock_is_real"] = False

        return values

    @validator("mock_is_real")
    def __mock_is_real_for_empty_mock_must_be_false(
        cls, v: bool, values: dict[str, Any], **kwargs: Any
    ) -> bool:
        if v and ((mock := values.get("mock")) is None or _is_action_data_empty(mock)):
            raise ValueError("mock_is_real must be False if mock is not provided")

        return v

    def add_data_subject(self, data_subject: DataSubject) -> None:
        self.data_subjects.append(data_subject)

    def add_contributor(
        self,
        name: str,
        email: str,
        role: Optional[Union[Enum, str]] = None,
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

    def set_mock(self, mock_data: Any, mock_is_real: bool) -> None:
        if isinstance(mock_data, SyftError):
            raise SyftException(mock_data)

        current_mock = self.mock
        self.mock = mock_data

        try:
            self.mock_is_real = mock_is_real
        except ValidationError as e:
            self.mock = current_mock
            raise e

    def no_mock(self) -> None:
        # relative
        from ..action.action_object import ActionObject

        self.mock = ActionObject.empty()

    def set_shape(self, shape: Tuple) -> None:
        self.shape = shape

    def check(self) -> Union[SyftSuccess, SyftError]:
        if not check_mock(self.data, self.mock):
            return SyftError(
                message=f"set_obj type {type(self.data)} must match set_mock type {type(self.mock)}"
            )
        if not _is_action_data_empty(self.mock):
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
    created_at: DateTime = DateTime.now()

    __attr_searchable__ = ["name", "citation", "url", "description", "action_ids"]
    __attr_unique__ = ["name"]
    __attr_repr_cols__ = ["name", "url", "created_at"]

    def _repr_html_(self) -> Any:
        uploaded_by_line = ""
        if len(self.contributors) > 0:
            uploaded_by_line = (
                f"<p><strong>Uploaded by: </strong>{self.contributors[0].name}</p>"
            )
        return (
            f"""
            <style>
            {fonts_css}
            .syft-dataset {{color: {SURFACE[options.color_theme]};}}
            .syft-dataset h3,
            .syft-dataset p
              {{font-family: 'Open Sans';}}
              {ITABLES_CSS}
            </style>
            """
            + "<div class='syft-dataset'>"
            + f"<h3>{self.name}</h3>"
            + f"<p>{self.description}</p>"
            + uploaded_by_line
            + f"<p><strong>Created on: </strong>{self.created_at}</p>"
            + f'<p><strong>URL: </strong><a href="{self.url}">{self.url}</a></p>'
            + "<p><strong>Contributors: </strong> to see full details call dataset.contributors</p>"
            + self.asset_list._repr_html_()
            + "</div>"
        )

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

    def _old_repr_markdown_(self) -> str:
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
        return as_markdown_python_code(_repr_str)

    def _repr_markdown_(self) -> str:
        # return self._old_repr_markdown_()
        return self._markdown_()

    def _markdown_(self) -> str:
        _repr_str = f"Syft Dataset: {self.name}\n\n"
        _repr_str += "Assets:\n\n"
        for asset in self.asset_list:
            _repr_str += f"\t{asset.name}: {asset.description}\n\n"
        if self.citation:
            _repr_str += f"Citation: {self.citation}\n\n"
        if self.url:
            _repr_str += f"URL: {self.url}\n\n"
        if self.description:
            _repr_str += f"Description: \n\n{self.description}\n\n"
        return _repr_str

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


_ASSET_WITH_NONE_MOCK_ERROR_MESSAGE: str = "".join(
    [
        "To be included in a Dataset, an asset must either contain a mock, ",
        "or have it explicitly set to be empty.\n",
        "You can create an asset without a mock with `sy.Asset(..., mock=sy.ActionObject.empty())` or\n"
        "set the mock of an existing asset to be empty with `asset.no_mock()` or ",
        "`asset.mock = sy.ActionObject.empty()`.",
    ]
)


def _check_asset_must_contain_mock(asset_list: List[CreateAsset]) -> None:
    assets_without_mock = [asset.name for asset in asset_list if asset.mock is None]
    if assets_without_mock:
        raise ValueError(
            "".join(
                [
                    "These assets do not contain a mock:\n",
                    *[f"{asset}\n" for asset in assets_without_mock],
                    "\n",
                    _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE,
                ]
            )
        )


@serializable()
class CreateDataset(Dataset):
    # version
    __canonical_name__ = "CreateDataset"
    __version__ = SYFT_OBJECT_VERSION_1
    asset_list: List[CreateAsset] = []

    id: Optional[UID] = None
    created_at: Optional[DateTime]

    class Config:
        validate_assignment = True

    def _check_asset_must_contain_mock(self) -> None:
        _check_asset_must_contain_mock(self.asset_list)

    @validator("asset_list")
    def __assets_must_contain_mock(
        cls, asset_list: List[CreateAsset]
    ) -> List[CreateAsset]:
        _check_asset_must_contain_mock(asset_list)
        return asset_list

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
        role: Optional[Union[Enum, str]] = None,
        phone: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        _role_str = role.value if isinstance(role, Enum) else role
        contributor = Contributor(
            name=name, role=_role_str, email=email, phone=phone, note=note
        )
        self.contributors.append(contributor)

    def add_asset(self, asset: CreateAsset, force_replace=False) -> None:
        if asset.mock is None:
            raise ValueError(_ASSET_WITH_NONE_MOCK_ERROR_MESSAGE)

        for i, existing_asset in enumerate(self.asset_list):
            if existing_asset.name == asset.name:
                if not force_replace:
                    print(
                        f'WARNING: Asset "{asset.name}" already exists. To replace it,\
                            use add_asset(asset, force_replace=True)'
                    )
                    return
                else:
                    self.asset_list[i] = asset
                    print(f"Asset {asset.name} has been successfully replaced.")
                    return

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
        if not _is_action_data_empty(context.obj.mock):
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


def add_msg_creation_time(context: TransformContext) -> TransformContext:
    context.output["created_at"] = DateTime.now()
    return context


@transform(CreateAsset, Asset)
def createasset_to_asset() -> List[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        infer_shape,
        create_and_store_twin,
        set_data_subjects,
    ]


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
    return [
        generate_id,
        add_msg_creation_time,
        validate_url,
        convert_asset,
        add_current_date,
    ]


class DatasetUpdate:
    pass
