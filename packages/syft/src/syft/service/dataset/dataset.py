# stdlib
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

# third party
from IPython.display import HTML
from IPython.display import display
import itables
import pandas as pd
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
from ...types.dicttuple import DictTuple
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
from ...util.notebook_ui.notebook_addons import FOLDER_ICON
from ...util.util import get_mb_size
from ..data_subject.data_subject import DataSubject
from ..data_subject.data_subject import DataSubjectCreate
from ..data_subject.data_subject_service import DataSubjectService
from ..response import SyftError
from ..response import SyftException
from ..response import SyftSuccess

DATA_SIZE_WARNING_LIMIT = 512


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

    __repr_attrs__ = ["name", "role", "email"]

    def _repr_html_(self) -> Any:
        return f"""
            <style>
            .syft-contributor {{color: {SURFACE[options.color_theme]};}}
            </style>
            <div class='syft-contributor' style="line-height:25%">
                <h3>Contributor</h3>
                <p><strong>Name: </strong>{self.name}</p>
                <p><strong>Role: </strong>{self.role}</p>
                <p><strong>Email: </strong>{self.email}</p>
            </div>
            """

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Contributor):
            return False

        # We assoctiate two contributors as equal if they have the same email
        return self.email == value.email

    def __hash__(self) -> int:
        return hash(self.email)


@serializable()
class MarkdownDescription(SyftObject):
    # version
    __canonical_name__ = "MarkdownDescription"
    __version__ = SYFT_OBJECT_VERSION_1

    text: str

    def _repr_markdown_(self) -> str:
        style = """
        <style>
            .jp-RenderedHTMLCommon pre {
                background-color: #282c34 !important;
                padding: 10px 10px 10px;
            }
            .jp-RenderedHTMLCommon pre code {
                background-color: #282c34 !important;  /* Set the background color for the text in the code block */
                color: #abb2bf !important;  /* Set text color */
            }
        </style>
        """
        display(HTML(style))
        return self.text


@serializable()
class Asset(SyftObject):
    # version
    __canonical_name__ = "Asset"
    __version__ = SYFT_OBJECT_VERSION_1

    action_id: UID
    node_uid: UID
    name: str
    description: Optional[MarkdownDescription] = None
    contributors: Set[Contributor] = set()
    data_subjects: List[DataSubject] = []
    mock_is_real: bool = False
    shape: Optional[Tuple]
    created_at: DateTime = DateTime.now()
    uploader: Optional[Contributor]

    __repr_attrs__ = ["name", "shape"]

    def __init__(
        self,
        description: Optional[Union[MarkdownDescription, str]] = "",
        **data: Any,
    ):
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**data, description=description)

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

        uploaded_by_line = (
            f"<p><strong>Uploaded by: </strong>{self.uploader.name} ({self.uploader.email})</p>"
            if self.uploader
            else ""
        )

        if isinstance(self.data, ActionObject):
            data_table_line = itables.to_html_datatable(
                df=self.data.syft_action_data, css=itables_css
            )
        elif isinstance(self.data, pd.DataFrame):
            data_table_line = itables.to_html_datatable(df=self.data, css=itables_css)
        else:
            data_table_line = self.data

        if isinstance(self.mock, ActionObject):
            mock_table_line = itables.to_html_datatable(
                df=self.mock.syft_action_data, css=itables_css
            )
        elif isinstance(self.mock, pd.DataFrame):
            mock_table_line = itables.to_html_datatable(df=self.mock, css=itables_css)
        else:
            mock_table_line = self.mock
            if isinstance(mock_table_line, SyftError):
                mock_table_line = mock_table_line.message

        return f"""
            <style>
            {fonts_css}
            .syft-asset {{color: {SURFACE[options.color_theme]};}}
            .syft-asset h3,
            .syft-asset p
              {{font-family: 'Open Sans'}}
            {ITABLES_CSS}
            </style>

            <div class="syft-asset">
            <h3>{self.name}</h3>
            <p>{self.description}</p>
            <p><strong>Asset ID: </strong>{self.id}</p>
            <p><strong>Action Object ID: </strong>{self.action_id}</p>
            {uploaded_by_line}
            <p><strong>Created on: </strong>{self.created_at}</p>
            <p><strong>Data:</strong></p>
            {data_table_line}
            <p><strong>Mock Data:</strong></p>
            {mock_table_line}
            </div>"""

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Asset):
            return False
        return (
            self.action_id == other.action_id
            and self.name == other.name
            and self.contributors == other.contributors
            and self.shape == other.shape
            and self.description == other.description
            and self.data_subjects == other.data_subjects
            and self.mock_is_real == other.mock_is_real
            and self.uploader == other.uploader
            and self.created_at == other.created_at
        )

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
    def mock(self) -> Union[SyftError, Any]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        result = api.services.action.get_mock(self.action_id)
        try:
            if isinstance(result, SyftObject):
                return result.syft_action_data
            return result
        except Exception as e:
            return SyftError(message=f"Failed to get mock. {e}")

    def has_data_permission(self) -> bool:
        return self.data is not None

    def has_permission(self, data_result: Any) -> bool:
        # TODO: implement in a better way
        return not (
            isinstance(data_result, str)
            and data_result.startswith("Permission")
            and data_result.endswith("denied")
        )

    @property
    def data(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        res = api.services.action.get(self.action_id)
        if self.has_permission(res):
            return res.syft_action_data
        else:
            return None


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
    description: Optional[MarkdownDescription] = None
    contributors: Set[Contributor] = set()
    data_subjects: List[DataSubjectCreate] = []
    node_uid: Optional[UID]
    action_id: Optional[UID]
    data: Optional[Any]
    mock: Optional[Any]
    shape: Optional[Tuple]
    mock_is_real: bool = False
    created_at: Optional[DateTime]
    uploader: Optional[Contributor]

    __repr_attrs__ = ["name"]

    class Config:
        validate_assignment = True

    def __init__(self, description: Optional[str] = "", **data: Any) -> None:
        super().__init__(**data, description=MarkdownDescription(text=str(description)))

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
    ) -> Union[SyftSuccess, SyftError]:
        try:
            _role_str = role.value if isinstance(role, Enum) else role
            contributor = Contributor(
                name=name, role=_role_str, email=email, phone=phone, note=note
            )
            if contributor in self.contributors:
                return SyftError(
                    message=f"Contributor with email: '{email}' already exists in '{self.name}' Asset."
                )
            self.contributors.add(contributor)

            return SyftSuccess(
                message=f"Contributor '{name}' added to '{self.name}' Asset."
            )
        except Exception as e:
            return SyftError(message=f"Failed to add contributor. Error: {e}")

    def set_description(self, description: str) -> None:
        self.description = MarkdownDescription(text=description)

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

        self.set_mock(ActionObject.empty(), False)

    def set_shape(self, shape: Tuple) -> None:
        self.shape = shape

    def check(self) -> Union[SyftSuccess, SyftError]:
        if not check_mock(self.data, self.mock):
            return SyftError(
                message=f"set_obj type {type(self.data)} must match set_mock type {type(self.mock)}"
            )
        # if not _is_action_data_empty(self.mock):
        #     data_shape = get_shape_or_len(self.data)
        #     mock_shape = get_shape_or_len(self.mock)
        #     if data_shape != mock_shape:
        #         return SyftError(
        #             message=f"set_obj shape {data_shape} must match set_mock shape {mock_shape}"
        #         )
        total_size_mb = get_mb_size(self.data) + get_mb_size(self.mock)
        if total_size_mb > DATA_SIZE_WARNING_LIMIT:
            print(
                f"**WARNING**: The total size for asset: '{self.name}' exceeds '{DATA_SIZE_WARNING_LIMIT} MB'. "
                "This might result in failure to upload dataset. "
                "Please contact #support on OpenMined slack for further assistance.",
            )

        return SyftSuccess(message="Dataset is Valid")


def get_shape_or_len(obj: Any) -> Optional[Union[Tuple[int, ...], int]]:
    if hasattr(obj, "shape"):
        shape = getattr(obj, "shape", None)
        if shape:
            return shape
    len_attr = getattr(obj, "__len__", None)
    if len_attr is not None:
        len_value = len_attr()
        if isinstance(len_value, int):
            return (len_value,)
        return len_value
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
    contributors: Set[Contributor] = set()
    citation: Optional[str]
    url: Optional[str]
    description: Optional[MarkdownDescription] = None
    updated_at: Optional[str]
    requests: Optional[int] = 0
    mb_size: Optional[int]
    created_at: DateTime = DateTime.now()
    uploader: Contributor

    __attr_searchable__ = ["name", "citation", "url", "description", "action_ids"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "url", "created_at"]

    def __init__(
        self,
        description: Optional[Union[str, MarkdownDescription]] = "",
        **data: Any,
    ) -> None:
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**data, description=description)

    @property
    def icon(self) -> str:
        return FOLDER_ICON

    def _coll_repr_(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Assets": len(self.asset_list),
            "Size": f"{self.mb_size} (MB)",
            "Url": self.url,
            "created at": str(self.created_at),
        }

    def _repr_html_(self) -> Any:
        uploaded_by_line = (
            (
                "<p class='paragraph-sm'><strong>"
                + f"<span class='pr-8'>Uploaded by:</span></strong>{self.uploader.name} ({self.uploader.email})</p>"
            )
            if self.uploader
            else ""
        )
        description_text: str = self.description.text if self.description else ""
        return f"""
            <style>
            {fonts_css}
            .syft-dataset {{color: {SURFACE[options.color_theme]};}}
            .syft-dataset h3,
            .syft-dataset p
              {{font-family: 'Open Sans';}}
              {ITABLES_CSS}
            </style>
            <div class='syft-dataset'>
            <h3>{self.name}</h3>
            <p>{description_text}</p>
            {uploaded_by_line}
            <p class='paragraph-sm'><strong><span class='pr-8'>Created on: </span></strong>{self.created_at}</p>
            <p class='paragraph-sm'><strong><span class='pr-8'>URL:
            </span></strong><a href='{self.url}'>{self.url}</a></p>
            <p class='paragraph-sm'><strong><span class='pr-8'>Contributors:</span></strong>
            to see full details call <strong>dataset.contributors</strong></p>
            {self.assets._repr_html_()}
            """

    def action_ids(self) -> List[UID]:
        data = []
        for asset in self.asset_list:
            if asset.action_id:
                data.append(asset.action_id)
        return data

    @property
    def assets(self) -> DictTuple[str, Asset]:
        return DictTuple((asset.name, asset) for asset in self.asset_list)

    def _old_repr_markdown_(self) -> str:
        _repr_str = f"Syft Dataset: {self.name}\n"
        _repr_str += "Assets:\n"
        for asset in self.asset_list:
            if asset.description is not None:
                _repr_str += f"\t{asset.name}: {asset.description.text}\n\n"
            else:
                _repr_str += f"\t{asset.name}\n\n"
        if self.citation:
            _repr_str += f"Citation: {self.citation}\n"
        if self.url:
            _repr_str += f"URL: {self.url}\n"
        if self.description:
            _repr_str += f"Description: {self.description.text}\n"
        return as_markdown_python_code(_repr_str)

    def _repr_markdown_(self) -> str:
        # return self._old_repr_markdown_()
        return self._markdown_()

    def _markdown_(self) -> str:
        _repr_str = f"Syft Dataset: {self.name}\n\n"
        _repr_str += "Assets:\n\n"
        for asset in self.asset_list:
            if asset.description is not None:
                _repr_str += f"\t{asset.name}: {asset.description.text}\n\n"
            else:
                _repr_str += f"\t{asset.name}\n\n"
        if self.citation:
            _repr_str += f"Citation: {self.citation}\n\n"
        if self.url:
            _repr_str += f"URL: {self.url}\n\n"
        if self.description:
            _repr_str += f"Description: \n\n{self.description.text}\n\n"
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
class DatasetPageView(SyftObject):
    # version
    __canonical_name__ = "DatasetPageView"
    __version__ = SYFT_OBJECT_VERSION_1

    datasets: DictTuple
    total: int


@serializable()
class CreateDataset(Dataset):
    # version
    __canonical_name__ = "CreateDataset"
    __version__ = SYFT_OBJECT_VERSION_1
    asset_list: List[CreateAsset] = []

    __repr_attrs__ = ["name", "url"]

    id: Optional[UID] = None
    created_at: Optional[DateTime]
    uploader: Optional[Contributor]  # type: ignore[assignment]

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
        self.description = MarkdownDescription(text=description)

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
    ) -> Union[SyftSuccess, SyftError]:
        try:
            _role_str = role.value if isinstance(role, Enum) else role
            contributor = Contributor(
                name=name, role=_role_str, email=email, phone=phone, note=note
            )
            if contributor in self.contributors:
                return SyftError(
                    message=f"Contributor with email: '{email}' already exists in '{self.name}' Dataset."
                )
            self.contributors.add(contributor)
            return SyftSuccess(
                message=f"Contributor '{name}' added to '{self.name}' Dataset."
            )
        except Exception as e:
            return SyftError(message=f"Failed to add contributor. Error: {e}")

    def add_asset(
        self, asset: CreateAsset, force_replace: bool = False
    ) -> Union[SyftSuccess, SyftError]:
        if asset.mock is None:
            raise ValueError(_ASSET_WITH_NONE_MOCK_ERROR_MESSAGE)

        for i, existing_asset in enumerate(self.asset_list):
            if existing_asset.name == asset.name:
                if not force_replace:
                    return SyftError(
                        message=f"""Asset "{asset.name}" already exists in '{self.name}' Dataset."""
                        """ Use add_asset(asset, force_replace=True) to replace."""
                    )
                else:
                    self.asset_list[i] = asset
                    return SyftSuccess(
                        f"Asset {asset.name} has been successfully replaced."
                    )

        self.asset_list.append(asset)

        return SyftSuccess(
            message=f"Asset '{asset.name}' added to '{self.name}' Dataset."
        )

    def replace_asset(self, asset: CreateAsset) -> Union[SyftSuccess, SyftError]:
        return self.add_asset(asset=asset, force_replace=True)

    def remove_asset(self, name: str) -> None:
        asset_to_remove = None
        for asset in self.asset_list:
            if asset.name == name:
                asset_to_remove = asset
                break

        if asset_to_remove is None:
            return SyftError(message=f"No asset exists with name: {name}")
        self.asset_list.remove(asset_to_remove)
        return SyftSuccess(
            message=f"Asset '{self.name}' removed from '{self.name}' Dataset."
        )

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


def add_default_node_uid(context: TransformContext) -> TransformContext:
    if context.output["node_uid"] is None:
        context.output["node_uid"] = context.node.id
    return context


@transform(CreateAsset, Asset)
def createasset_to_asset() -> List[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        infer_shape,
        create_and_store_twin,
        set_data_subjects,
        add_default_node_uid,
    ]


def convert_asset(context: TransformContext) -> TransformContext:
    assets = context.output.pop("asset_list", [])
    for idx, create_asset in enumerate(assets):
        asset_context = TransformContext.from_context(obj=create_asset, context=context)
        assets[idx] = create_asset.to(Asset, context=asset_context)
    context.output["asset_list"] = assets
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
