# stdlib
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

# third party
from pydantic import ConfigDict

# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...types.datetime import DateTime
from ...types.dicttuple import DictTuple
from ...types.markdown import MarkdownDescription
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.transforms import validate_url
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.notebook_ui.notebook_addons import FOLDER_ICON
from ..response import SyftError
from ..response import SyftException
from ..response import SyftSuccess

DATA_SIZE_WARNING_LIMIT = 512


NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable()
class ModelContributor(SyftObject):
    __canonical_name__ = "ModelContributor"
    __version__ = SYFT_OBJECT_VERSION_2

    name: str
    role: str | None = None
    email: str
    phone: str | None = None
    note: str | None = None

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
        if not isinstance(value, ModelContributor):
            return False

        # We assoctiate two contributors as equal if they have the same email
        return self.email == value.email

    def __hash__(self) -> int:
        return hash(self.email)


@serializable()
class ModelEndpoint(SyftObject):
    # version
    __canonical_name__ = "ModelEndpoint"
    __version__ = SYFT_OBJECT_VERSION_2

    action_id: UID
    node_uid: UID
    name: str
    path: str
    description: MarkdownDescription | None = None
    contributors: set[ModelContributor] = set()
    created_at: DateTime = DateTime.now()
    uploader: ModelContributor | None = None

    __repr_attrs__ = ["name", "shape"]

    def __init__(
        self,
        description: MarkdownDescription | str | None = "",
        **data: Any,
    ):
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**data, description=description)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelEndpoint):
            return False
        return (
            self.action_id == other.action_id
            and self.name == other.name
            and self.contributors == other.contributors
            and self.path == other.path
            and self.description == other.description
            and self.uploader == other.uploader
            and self.created_at == other.created_at
        )


@serializable()
class CreateModelEndpoint(SyftObject):
    # version
    __canonical_name__ = "CreateModelEndpoint"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID | None = None  # type:ignore[assignment]
    name: str
    description: MarkdownDescription | None = None
    contributors: set[ModelContributor] = set()
    node_uid: UID | None = None
    action_id: UID | None = None
    created_at: DateTime | None = None
    uploader: ModelContributor | None = None

    __repr_attrs__ = ["name"]
    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, description: str | None = "", **data: Any) -> None:
        super().__init__(**data, description=MarkdownDescription(text=str(description)))

    def add_contributor(
        self,
        name: str,
        email: str,
        role: Enum | str | None = None,
        phone: str | None = None,
        note: str | None = None,
    ) -> SyftSuccess | SyftError:
        try:
            _role_str = role.value if isinstance(role, Enum) else role
            contributor = ModelContributor(
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


@serializable()
class Model(SyftObject):
    # version
    __canonical_name__: str = "Model"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    name: str
    node_uid: UID | None = None
    endpoint_list: list[ModelEndpoint] = []
    contributors: set[ModelContributor] = set()
    citation: str | None = None
    url: str | None = None
    description: MarkdownDescription | None = None
    updated_at: str | None = None
    created_at: DateTime = DateTime.now()
    uploader: ModelContributor

    __attr_searchable__ = ["name", "citation", "url", "description", "action_ids"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "url", "created_at"]

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
        return FOLDER_ICON

    @property
    def endpoints(self) -> DictTuple[str, ModelEndpoint]:
        return DictTuple((endpoint.name, endpoint) for endpoint in self.endpoint_list)


@serializable()
class CreateModel(Model):
    # version
    __canonical_name__ = "CreateDataset"
    __version__ = SYFT_OBJECT_VERSION_2
    endpoint_list: list[CreateModelEndpoint] = []

    __repr_attrs__ = ["name", "url"]

    id: UID | None = None  # type: ignore[assignment]
    created_at: DateTime | None = None  # type: ignore[assignment]
    uploader: ModelContributor | None = None  # type: ignore[assignment]

    model_config = ConfigDict(validate_assignment=True)

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
        role: Enum | str | None = None,
        phone: str | None = None,
        note: str | None = None,
    ) -> SyftSuccess | SyftError:
        try:
            _role_str = role.value if isinstance(role, Enum) else role
            contributor = ModelContributor(
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

    def add_endpoint(
        self, endpoint: ModelEndpoint, force_replace: bool = False
    ) -> SyftSuccess | SyftError:
        for i, existing_endpoint in enumerate(self.endpoint_list):
            if existing_endpoint.name == endpoint.name:
                if not force_replace:
                    return SyftError(
                        message=f"""Endpoint "{endpoint.name}" already exists in '{self.name}' Model."""
                        """ Use add_endpoint(endpoint, force_replace=True) to replace."""
                    )
                else:
                    self.endpoint_list[i] = endpoint
                    return SyftSuccess(
                        f"Endpoint {endpoint.name} has been successfully replaced."
                    )

        self.endpoint_list.append(endpoint)

        return SyftSuccess(
            message=f"Endpoint '{endpoint.name}' added to '{self.name}' Model."
        )

    def replace_asset(self, endpoint: CreateModelEndpoint) -> SyftSuccess | SyftError:
        return self.add_endpoint(endpoint=endpoint, force_replace=True)


def add_msg_creation_time(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    context.output["created_at"] = DateTime.now()
    return context


def add_default_node_uid(context: TransformContext) -> TransformContext:
    if context.output is not None:
        if context.output["node_uid"] is None and context.node is not None:
            context.output["node_uid"] = context.node.id
    else:
        raise ValueError(f"{context}'s output is None. No transformation happened")
    return context


@transform(CreateModelEndpoint, ModelEndpoint)
def createasset_to_asset() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        add_default_node_uid,
    ]


def convert_model_endpoint(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    endpoints = context.output.pop("endpoint_list", [])
    for idx, create_model_endpoint in enumerate(endpoints):
        endpoint_context = TransformContext.from_context(
            obj=create_model_endpoint, context=context
        )
        endpoints[idx] = create_model_endpoint.to(
            ModelEndpoint, context=endpoint_context
        )
    context.output["endpoint_list"] = endpoints

    return context


def add_current_date(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    current_date = datetime.now()
    formatted_date = current_date.strftime("%b %d, %Y")
    context.output["updated_at"] = formatted_date

    return context


@transform(CreateModel, Model)
def createmodel_to_model() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        validate_url,
        convert_model_endpoint,
        add_current_date,
    ]
