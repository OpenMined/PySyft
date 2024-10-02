# stdlib
from collections.abc import Callable
from datetime import datetime
from enum import Enum
import logging
import textwrap
from typing import Any

# third party
from IPython.display import display
import markdown
import pandas as pd
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ...types.dicttuple import DictTuple
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.transforms import validate_url
from ...types.uid import UID
from ...util.markdown import as_markdown_python_code
from ...util.misc_objs import MarkdownDescription
from ...util.notebook_ui.icons import Icon
from ...util.table import itable_template_from_df
from ...util.util import repr_truncation
from ..action.action_data_empty import ActionDataEmpty
from ..action.action_object import ActionObject
from ..data_subject.data_subject import DataSubject
from ..data_subject.data_subject import DataSubjectCreate
from ..response import SyftError
from ..response import SyftSuccess
from ..response import SyftWarning

logger = logging.getLogger(__name__)


@serializable()
class Contributor(SyftObject):
    __canonical_name__ = "Contributor"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    role: str | None = None
    email: str
    phone: str | None = None
    note: str | None = None

    __repr_attrs__ = ["name", "role", "email"]

    def _repr_html_(self) -> Any:
        return f"""

                Contributor
                Name: {self.name}
                Role: {self.role}
                Email: {self.email}

            """

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Contributor):
            return False

        # We assoctiate two contributors as equal if they have the same email
        return self.email == value.email

    def __hash__(self) -> int:
        return hash(self.email)


@serializable()
class Asset(SyftObject):
    # version
    __canonical_name__ = "Asset"
    __version__ = SYFT_OBJECT_VERSION_1

    action_id: UID
    server_uid: UID
    name: str
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    data_subjects: list[DataSubject] = []
    mock_is_real: bool = False
    shape: tuple | None = None
    created_at: DateTime = DateTime.now()
    uploader: Contributor | None = None

    # _kwarg_name and _dataset_name are set by the UserCode.assets
    _kwarg_name: str | None = None
    _dataset_name: str | None = None
    __syft_include_id_coll_repr__ = False

    def __init__(
        self,
        description: MarkdownDescription | str | None = "",
        **data: Any,
    ):
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**data, description=description)

    def _repr_html_(self) -> Any:
        # relative
        from ...service.action.action_object import ActionObject

        uploaded_by_line = (
            f"Uploaded by: {self.uploader.name} ({self.uploader.email})"
            if self.uploader
            else ""
        )

        mock = self.mock
        private_data_res = self._private_data()
        if private_data_res.is_err():
            data_table_line = "You have no permission to the private data"
        else:
            private_data_obj = private_data_res.ok()
            if isinstance(private_data_obj, ActionObject):
                if isinstance(private_data_obj.syft_action_data, ActionDataEmpty):
                    data_table_line = "No data"
                else:
                    df = pd.DataFrame(private_data_obj)
                    data_table_line = itable_template_from_df(
                        df=private_data_obj.head(5)
                    )

            elif isinstance(private_data_obj, pd.DataFrame):
                data_table_line = itable_template_from_df(df=private_data_obj.head(5))
            else:
                try:
                    data_table_line = repr_truncation(private_data_obj)
                except Exception as e:
                    error_msg = (
                        e.public_message if isinstance(e, SyftException) else str(e)
                    )
                    logger.debug(f"Failed to truncate private data repr. {error_msg}")
                    data_table_line = private_data_res.ok()  # type: ignore

        if isinstance(mock, ActionObject):
            if isinstance(mock.syft_action_data, ActionDataEmpty):
                mock_table_line = "No data"
            else:
                df = pd.DataFrame(mock.syft_action_data)
                mock_table_line = itable_template_from_df(df=df.head(5))
        elif isinstance(mock, pd.DataFrame):
            mock_table_line = itable_template_from_df(df=self.mock.head(5))
        else:
            try:
                mock_table_line = repr_truncation(self.mock)
            except Exception as e:
                logger.debug(f"Failed to truncate mock data repr. {e}")
                mock_table_line = self.mock

            if isinstance(mock_table_line, SyftError):
                mock_table_line = mock_table_line.message

        return f"""
            <div class="syft-asset">
            <h3>{self.name}</h3>
            <p>{self.description or ""}</p>
            <p><strong>Asset ID: </strong>{self.id}</p>
            <p><strong>Action Object ID: </strong>{self.action_id}</p>
            {uploaded_by_line}
            <p><strong>Created on: </strong>{self.created_at}</p>
            <p><strong>Data:</strong></p>
            {data_table_line}
            <p><strong>Mock Data:</strong></p>
            {mock_table_line}
            </div>"""

    def __repr__(self) -> str:
        return f"Asset(name='{self.name}', server_uid='{self.server_uid}', action_id='{self.action_id}')"

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _repr_str = f"Asset: {self.name}\n"
        _repr_str += f"Pointer Id: {self.action_id}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Total Data Subjects: {len(self.data_subjects)}\n"
        _repr_str += f"Shape: {self.shape}\n"
        _repr_str += f"Contributors: {len(self.contributors)}\n"
        for contributor in self.contributors:
            _repr_str += f"\t{contributor.name}: {contributor.email}\n"
        return as_markdown_python_code(_repr_str)

    def _coll_repr_(self) -> dict[str, Any]:
        base_dict = {
            "Parameter": self._kwarg_name,
            "Action ID": self.action_id,
            "Asset Name": self.name,
            "Dataset Name": self._dataset_name,
            "Server UID": self.server_uid,
        }

        # _kwarg_name and _dataset_name are set by the UserCode.assets
        # if they are None, we remove them from the dict
        filtered_dict = {
            key: value for key, value in base_dict.items() if value is not None
        }
        return filtered_dict

    def _get_dict_for_user_code_repr(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id.no_dash,
            "source_asset": self.name,
            "source_dataset": self._dataset_name,
            "source_server": self.server_uid.no_dash,
        }

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
        api = self.get_api()
        if api.services is not None:
            return api.services.action.get_pointer(self.action_id)

    @property
    def mock(self) -> Any:
        # relative
        api = self.get_api()
        try:
            result = api.services.action.get_mock(self.action_id)
            if isinstance(result, SyftObject):
                return result.syft_action_data
            else:
                return result
        except Exception as e:
            raise SyftException.from_exception(
                e, public_message=f"Failed to get mock. {e}"
            )

    def has_data_permission(self) -> bool:
        return self.data is not None

    def has_permission(self, data_result: Any) -> bool:
        # TODO: implement in a better way
        return not (
            isinstance(data_result, str)
            and data_result.startswith("Permission")
            and data_result.endswith("denied")
        )

    @as_result(SyftException)
    def _private_data(self) -> Any:
        """
        Retrieves the private data associated with this asset.

        Returns:
            Result[Any, str]: A Result object containing the private data if the user has permission
            otherwise an Err object with the message "You do not have permission to access private data."
        """

        # TODO: split this out in permission logic and existence logic
        api = self.get_api_wrapped()
        if api.is_err():
            return None
        res = api.unwrap().services.action.get(self.action_id)
        if self.has_permission(res):
            return res.syft_action_data
        else:
            raise SyftException(public_message="You have no access to the private data")

    @property
    def data(self) -> Any:
        try:
            return self._private_data().unwrap()
        except SyftException:
            display(SyftError(message="You have no access to the private data"))
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

    return _is_action_data_empty(mock) or _is_action_data_empty(data)


@serializable()
class CreateAsset(SyftObject):
    # version
    __canonical_name__ = "CreateAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type:ignore[assignment]
    name: str
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    data_subjects: list[DataSubjectCreate] = []
    server_uid: UID | None = None
    action_id: UID | None = None
    data: Any | None = None
    mock: Any | None = None
    shape: tuple | None = None
    mock_is_real: bool = False
    created_at: DateTime | None = None
    uploader: Contributor | None = None

    __repr_attrs__ = ["name"]
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def __init__(self, description: str | None = None, **data: Any) -> None:
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**data, description=description)

    @model_validator(mode="after")
    def __mock_is_real_for_empty_mock_must_be_false(self) -> Self:
        if self.mock_is_real and (
            self.mock is None or _is_action_data_empty(self.mock)
        ):
            self.__dict__["mock_is_real"] = False

        return self

    def contains_empty(self) -> bool:
        if isinstance(self.mock, ActionObject) and isinstance(
            self.mock.syft_action_data_cache, ActionDataEmpty
        ):
            return True
        if isinstance(self.data, ActionObject) and isinstance(
            self.data.syft_action_data_cache, ActionDataEmpty
        ):
            return True
        return False

    def add_data_subject(self, data_subject: DataSubject) -> None:
        self.data_subjects.append(data_subject)

    def add_contributor(
        self,
        name: str,
        email: str,
        role: Enum | str | None = None,
        phone: str | None = None,
        note: str | None = None,
    ) -> SyftSuccess:
        try:
            _role_str = role.value if isinstance(role, Enum) else role
            contributor = Contributor(
                name=name, role=_role_str, email=email, phone=phone, note=note
            )
            if contributor in self.contributors:
                raise SyftException(
                    public_message=f"Contributor with email: '{email}' already exists in '{self.name}' Asset."
                )
            self.contributors.add(contributor)

            return SyftSuccess(
                message=f"Contributor '{name}' added to '{self.name}' Asset."
            )
        except Exception as e:
            raise SyftException(public_message=f"Failed to add contributor. Error: {e}")

    def set_description(self, description: str) -> None:
        self.description = MarkdownDescription(text=description)

    def set_obj(self, data: Any) -> None:
        if isinstance(data, SyftError):
            raise SyftException(public_message=data)
        self.data = data

    def set_mock(self, mock_data: Any, mock_is_real: bool) -> None:
        if isinstance(mock_data, SyftError):
            raise SyftException(public_message=mock_data)

        if mock_is_real and (mock_data is None or _is_action_data_empty(mock_data)):
            raise SyftException(
                public_message="`mock_is_real` must be False if mock is empty"
            )

        self.mock = mock_data
        self.mock_is_real = mock_is_real

    def no_mock(self) -> None:
        # relative
        from ..action.action_object import ActionObject

        self.set_mock(ActionObject.empty(), False)

    def set_shape(self, shape: tuple) -> None:
        self.shape = shape

    def check(self) -> SyftSuccess:
        if not check_mock(self.data, self.mock):
            raise SyftException(
                public_message=f"set_obj type {type(self.data)} must match set_mock type {type(self.mock)}"
            )

        return SyftSuccess(message="Dataset is Valid")


def get_shape_or_len(obj: Any) -> tuple[int, ...] | int | None:
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
    __canonical_name__: str = "Dataset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    server_uid: UID | None = None
    asset_list: list[Asset] = []
    contributors: set[Contributor] = set()
    citation: str | None = None
    url: str | None = None
    description: MarkdownDescription | None = None
    updated_at: str | None = None
    requests: int | None = 0
    mb_size: float | None = None
    created_at: DateTime = DateTime.now()
    uploader: Contributor
    summary: str | None = None
    to_be_deleted: bool = False

    __attr_searchable__ = [
        "name",
        "citation",
        "url",
        "description",
        "action_ids",
        "summary",
    ]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "summary", "url", "created_at"]
    __table_sort_attr__ = "Created at"

    def __init__(
        self,
        description: str | MarkdownDescription | None = "",
        **data: Any,
    ) -> None:
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**data, description=description)

    @property
    def icon(self) -> str:
        return Icon.FOLDER.svg

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Summary": self.summary,
            "Assets": len(self.asset_list),
            "Size": f"{self.mb_size} (MB)",
            "Url": self.url,
            "Created at": str(self.created_at),
        }

    def _repr_html_(self) -> Any:
        uploaded_by_line = (
            (
                "<p class='paragraph-sm'><strong>"
                + f"<span class='pr-8'>Uploaded by: </span></strong>{self.uploader.name} ({self.uploader.email})</p>"
            )
            if self.uploader
            else ""
        )
        if self.description is not None and self.description.text:
            description_info_message = f"""
            <h2><strong><span class='pr-8'>Description</span></strong></h2>
            {markdown.markdown(self.description.text, extensions=["extra"])}
            """
        else:
            description_info_message = ""
        if self.to_be_deleted:
            return "This dataset has been marked for deletion. The underlying data may be not available."
        return f"""
            <div class='syft-dataset'>
            <h1>{self.name}</h1>
            <h2><strong><span class='pr-8'>Summary</span></strong></h2>
            {f"<p>{self.summary}</p>" if self.summary else ""}
            {description_info_message}
            <h2><strong><span class='pr-8'>Dataset Details</span></strong></h2>
            {uploaded_by_line}
            <p class='paragraph-sm'><strong><span class='pr-8'>Created on: </span></strong>{self.created_at}</p>
            <p class='paragraph-sm'><strong><span class='pr-8'>URL:
            </span></strong><a href='{self.url}'>{self.url}</a></p>
            <p class='paragraph-sm'><strong><span class='pr-8'>Contributors:</span></strong>
            To see full details call <strong>dataset.contributors</strong>.</p>
            <h2><strong><span class='pr-8'>Assets</span></strong></h2>
            {self.assets._repr_html_()}
            """

    @property
    def action_ids(self) -> list[UID]:
        return [asset.action_id for asset in self.asset_list if asset.action_id]

    @property
    def assets(self) -> DictTuple[str, Asset]:
        return DictTuple((asset.name, asset) for asset in self.asset_list)

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _repr_str = f"Syft Dataset: {self.name}\n\n"
        _repr_str += "Assets:\n\n"
        for asset in self.asset_list:
            if asset.description is not None:
                description_text = textwrap.shorten(
                    asset.description.text, width=100, placeholder="..."
                )
                _repr_str += f"\t{asset.name}: {description_text}\n\n"
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
    def client(self) -> Any | None:
        # relative
        from ...client.client import SyftClientSessionCache

        client = SyftClientSessionCache.get_client_for_server_uid(self.server_uid)
        if client is None:
            raise SyftException(
                public_message=f"No clients for {self.server_uid} in memory. Please login with sy.login"
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


def _check_asset_must_contain_mock(asset_list: list[CreateAsset]) -> None:
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
    __canonical_name__ = "DatasetPageView"
    __version__ = SYFT_OBJECT_VERSION_2

    datasets: DictTuple[str, Dataset]
    total: int


@serializable()
class DatasetPageViewV1(SyftObject):
    __canonical_name__ = "DatasetPageView"
    __version__ = SYFT_OBJECT_VERSION_1

    datasets: DictTuple
    total: int


@serializable()
class CreateDataset(Dataset):
    __canonical_name__ = "CreateDataset"
    __version__ = SYFT_OBJECT_VERSION_1
    asset_list: list[CreateAsset] = []

    __repr_attrs__ = ["name", "summary", "url"]

    id: UID | None = None  # type: ignore[assignment]
    created_at: DateTime | None = None  # type: ignore[assignment]
    uploader: Contributor | None = None  # type: ignore[assignment]

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @field_validator("asset_list")
    @classmethod
    def __assets_must_contain_mock(
        cls, asset_list: list[CreateAsset]
    ) -> list[CreateAsset]:
        _check_asset_must_contain_mock(asset_list)
        return asset_list

    @field_validator("to_be_deleted")
    @classmethod
    def __to_be_deleted_must_be_false(cls, v: bool) -> bool:
        if v is True:
            raise ValueError("to_be_deleted must be False")
        return v

    def set_description(self, description: str) -> None:
        self.description = MarkdownDescription(text=description)

    def set_summary(self, summary: str) -> None:
        self.summary = summary

    def add_citation(self, citation: str) -> None:
        self.citation = citation

    def add_url(self, url: str) -> None:
        self.url = url

    def add_contributor(
        self,
        name: str,
        email: str,
        role: Enum | str | None = None,
        phone: str | None = None,
        note: str | None = None,
    ) -> SyftSuccess:
        try:
            _role_str = role.value if isinstance(role, Enum) else role
            contributor = Contributor(
                name=name, role=_role_str, email=email, phone=phone, note=note
            )
            if contributor in self.contributors:
                raise SyftException(
                    public_message=f"Contributor with email: '{email}' already exists in '{self.name}' Dataset."
                )
            self.contributors.add(contributor)
            return SyftSuccess(
                message=f"Contributor '{name}' added to '{self.name}' Dataset."
            )
        except Exception as e:
            raise SyftException(public_message=f"Failed to add contributor. Error: {e}")

    def add_asset(self, asset: CreateAsset, force_replace: bool = False) -> SyftSuccess:
        if asset.mock is None:
            raise ValueError(_ASSET_WITH_NONE_MOCK_ERROR_MESSAGE)

        for i, existing_asset in enumerate(self.asset_list):
            if existing_asset.name == asset.name:
                if not force_replace:
                    raise SyftException(
                        public_message=(
                            f"Asset '{asset.name}' already exists in '{self.name}' Dataset."
                            "\nUse add_asset(asset, force_replace=True) to replace."
                        )
                    )
                else:
                    self.asset_list[i] = asset
                    return SyftSuccess(
                        message=f"Asset {asset.name} has been successfully replaced."
                    )

        self.asset_list.append(asset)

        return SyftSuccess(
            message=f"Asset '{asset.name}' added to '{self.name}' Dataset."
        )

    def replace_asset(self, asset: CreateAsset) -> SyftSuccess:
        return self.add_asset(asset=asset, force_replace=True)

    def remove_asset(self, name: str) -> SyftSuccess:
        asset_to_remove = None
        for asset in self.asset_list:
            if asset.name == name:
                asset_to_remove = asset
                break

        if asset_to_remove is None:
            raise SyftException(public_message=f"No asset exists with name: {name}")
        self.asset_list.remove(asset_to_remove)
        return SyftSuccess(
            message=f"Asset '{self.name}' removed from '{self.name}' Dataset."
        )

    def check(self) -> SyftSuccess:
        errors = []
        for asset in self.asset_list:
            result = asset.check()
            if not result:
                errors.append(result)
        if len(errors):
            raise SyftException(public_message=f"Errors: {errors}")
        return SyftSuccess(message="Dataset is Valid")


def create_and_store_twin(context: TransformContext) -> TransformContext:
    if context.output is None:
        raise ValueError(f"{context}'s output is None. No transformation happened")

    action_id = context.output["action_id"]
    if action_id is None:
        # relative
        from ...types.twin_object import TwinObject

        private_obj = context.output.pop("data", None)
        mock_obj = context.output.pop("mock", None)
        if private_obj is None and mock_obj is None:
            raise ValueError("No data and no action_id means this asset has no data")

        asset = context.obj  # type: ignore
        contains_empty = asset.contains_empty()  # type: ignore
        twin = TwinObject(
            private_obj=asset.data,  # type: ignore
            mock_obj=asset.mock,  # type: ignore
            syft_server_location=asset.syft_server_location,  # type: ignore
            syft_client_verify_key=asset.syft_client_verify_key,  # type: ignore
        )
        res = twin._save_to_blob_storage(allow_empty=contains_empty).unwrap()
        if isinstance(res, SyftWarning):
            logger.debug(res.message)
        # TODO, upload to blob storage here
        if context.server is None:
            raise ValueError(
                "f{context}'s server is None, please log in. No trasformation happened"
            )
        context.server.services.action._set(
            context=context.to_server_context(),
            action_object=twin,
        ).unwrap(public_message="Failed to create and store twin")
        context.output["action_id"] = twin.id
    else:
        private_obj = context.output.pop("data", None)
        mock_obj = context.output.pop("mock", None)

    return context


def infer_shape(context: TransformContext) -> TransformContext:
    if context.output is None:
        raise ValueError(f"{context}'s output is None. No transformation happened")
    if context.output["shape"] is None:
        if context.obj is not None and not _is_action_data_empty(context.obj.mock):
            context.output["shape"] = get_shape_or_len(context.obj.mock)
    return context


def set_data_subjects(context: TransformContext) -> TransformContext:
    if context.output is None:
        raise ValueError(f"{context}'s output is None. No transformation happened")
    if context.server is None:
        raise SyftException(
            public_message="f{context}'s server is None, please log in. No trasformation happened"
        )
    data_subjects = context.output["data_subjects"]
    resultant_data_subjects = []
    for data_subject in data_subjects:
        result = context.server.services.data_subject.get_by_name(
            context=context, name=data_subject.name
        )
        resultant_data_subjects.append(result)
    context.output["data_subjects"] = resultant_data_subjects
    return context


def add_msg_creation_time(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    context.output["created_at"] = DateTime.now()
    return context


def add_default_server_uid(context: TransformContext) -> TransformContext:
    if context.output is not None:
        if context.output["server_uid"] is None and context.server is not None:
            context.output["server_uid"] = context.server.id
    else:
        raise ValueError(f"{context}'s output is None. No transformation happened")
    return context


@transform(CreateAsset, Asset)
def createasset_to_asset() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        infer_shape,
        create_and_store_twin,
        set_data_subjects,
        add_default_server_uid,
    ]


def convert_asset(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    assets = context.output.pop("asset_list", [])
    for idx, create_asset in enumerate(assets):
        asset_context = TransformContext.from_context(obj=create_asset, context=context)
        assets[idx] = create_asset.to(Asset, context=asset_context)
    context.output["asset_list"] = assets

    return context


def add_current_date(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    current_date = datetime.now()
    formatted_date = current_date.strftime("%b %d, %Y")
    context.output["updated_at"] = formatted_date

    return context


@transform(CreateDataset, Dataset)
def createdataset_to_dataset() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        validate_url,
        convert_asset,
        add_current_date,
        make_set_default("to_be_deleted", False),  # explicitly set it to False
    ]


class DatasetUpdate(PartialSyftObject):
    __canonical_name__ = "DatasetUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    to_be_deleted: bool
