# stdlib
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

# third party
from IPython.display import HTML
from IPython.display import display
import itables
import pandas as pd
from pydantic import ConfigDict
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
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
from ...util.fonts import FONT_CSS
from ...util.fonts import ITABLES_CSS
from ...util.markdown import as_markdown_python_code
from ...util.notebook_ui.notebook_addons import FOLDER_ICON
from ..response import SyftError
from ..response import SyftSuccess
from ..dataset.dataset import MarkdownDescription
from ..dataset.dataset import Contributor



@serializable()
class ModelPageView(SyftObject):
    # version
    __canonical_name__ = "ModelPageView"
    __version__ = SYFT_OBJECT_VERSION_1

    models: DictTuple
    total: int
    
@serializable()
class ModelAsset(SyftObject):
    # version
    __canonical_name__ = "ModelAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    endpoint_path: str
    node_uid: UID
    created_at: DateTime = DateTime.now()

    __repr_attrs__ = ["name", "endpoint_path"]

    def __init__(
        self,
        description: MarkdownDescription | str | None = "",
        **kwargs: Any,
    ):
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**kwargs, description=description)

    def _repr_html_(self) -> Any:
        itables_css = f"""
        .itables table {{
            margin: 0 auto;
            float: left;
            color: {ON_SURFACE_HIGHEST[options.color_theme]};
        }}
        .itables table th {{color: {SURFACE_SURFACE[options.color_theme]};}}
        """

        return f"""
            <style>
            {FONT_CSS}
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
            <p><strong>Created on: </strong>{self.created_at}</p>
            </div>"""

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _repr_str = f"Asset: {self.name}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Contributors: {len(self.contributors)}\n"
        for contributor in self.contributors:
            _repr_str += f"\t{contributor.name}: {contributor.email}\n"
        return as_markdown_python_code(_repr_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelAsset):
            return False
        return (
            self.name == other.name
            and self.contributors == other.contributors
            and self.description == other.description
            and self.endpoint_path == other.endpoint_path
            and self.created_at == other.created_at
        )
    
    @property
    def endpoint(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is not None and api.services is not None:
            path_node = api.services
            for path in self.endpoint_path.split("."):
                try:
                    path_node = path_node.__getattribute__(path)
                except Exception as e:
                    print(f"Invalid path: {self.endpoint_path}")
                    raise e
            
            return path_node
    
    def __call__(self, *args, **kwargs) -> Any:
        endpoint = self.endpoint
        result = endpoint.__call__(*args, **kwargs)
        return result


@serializable()
class CreateModelAsset(SyftObject):
    # version
    __canonical_name__ = "CreateModelAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type:ignore[assignment]
    name: str
    node_uid: UID | None = None
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    endpoint_path: str | None = None
    created_at: DateTime | None = None

    __repr_attrs__ = ["name"]
    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, description: str | None = "", **kwargs: Any) -> None:
        super().__init__(**kwargs, description=MarkdownDescription(text=str(description)))

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

    def set_endpoint_path(self, path: str) -> None:
        self.endpoint_path = path


@serializable()
class Model(SyftObject):
    # version
    __canonical_name__: str = "Model"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    asset_list: list[Any] = []
    contributors: set[Contributor] = set()
    citation: str | None = None
    url: str | None = None
    description: MarkdownDescription | None = None
    updated_at: str | None = None
    created_at: DateTime = DateTime.now()

    __attr_searchable__ = ["name", "citation", "url", "description"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "url", "created_at"]

    def __init__(
        self,
        description: str | MarkdownDescription | None = "",
        **kwargs: Any,
    ) -> None:
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**kwargs, description=description)

    @property
    def icon(self) -> str:
        return FOLDER_ICON

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Assets": len(self.asset_list),
            "Url": self.url,
            "created at": str(self.created_at),
        }

    def _repr_html_(self) -> Any:
        description_text: str = self.description.text if self.description else ""
        return f"""
            <style>
            {FONT_CSS}
            .syft-dataset {{color: {SURFACE[options.color_theme]};}}
            .syft-dataset h3,
            .syft-dataset p
              {{font-family: 'Open Sans';}}
              {ITABLES_CSS}
            </style>
            <div class='syft-dataset'>
            <h3>{self.name}</h3>
            <p>{description_text}</p>
            <p class='paragraph-sm'><strong><span class='pr-8'>Created on: </span></strong>{self.created_at}</p>
            <p class='paragraph-sm'><strong><span class='pr-8'>URL:
            </span></strong><a href='{self.url}'>{self.url}</a></p>
            <p class='paragraph-sm'><strong><span class='pr-8'>Contributors:</span></strong>
            to see full details call <strong>dataset.contributors</strong></p>
            {self.assets._repr_html_()}
            """

    @property
    def assets(self) -> DictTuple[str, ModelAsset]:
        return DictTuple((asset.name, asset) for asset in self.asset_list)

    def _old_repr_markdown_(self) -> str:
        _repr_str = f"Syft Model: {self.name}\n"
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

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        # return self._old_repr_markdown_()
        return self._markdown_()

    def _markdown_(self) -> str:
        _repr_str = f"Syft Model: {self.name}\n\n"
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


@serializable()
class CreateModel(Model):
    # version
    __canonical_name__ = "CreateModel"
    __version__ = SYFT_OBJECT_VERSION_1
    asset_list: list[Any] = []

    __repr_attrs__ = ["name", "url"]

    id: UID | None = None  # type: ignore[assignment]
    created_at: DateTime | None = None  # type: ignore[assignment]

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
            contributor = Contributor(
                name=name, role=_role_str, email=email, phone=phone, note=note
            )
            if contributor in self.contributors:
                return SyftError(
                    message=f"Contributor with email: '{email}' already exists in '{self.name}' Model."
                )
            self.contributors.add(contributor)
            return SyftSuccess(
                message=f"Contributor '{name}' added to '{self.name}' Model."
            )
        except Exception as e:
            return SyftError(message=f"Failed to add contributor. Error: {e}")

    def add_asset(
        self, asset: CreateModelAsset, force_replace: bool = False
    ) -> SyftSuccess | SyftError:
        for i, existing_asset in enumerate(self.asset_list):
            if existing_asset.name == asset.name:
                if not force_replace:
                    return SyftError(
                        message=f"""Asset "{asset.name}" already exists in '{self.name}' Model."""
                        """ Use add_asset(asset, force_replace=True) to replace."""
                    )
                else:
                    self.asset_list[i] = asset
                    return SyftSuccess(
                        f"Asset {asset.name} has been successfully replaced."
                    )

        self.asset_list.append(asset)

        return SyftSuccess(
            message=f"Asset '{asset.name}' added to '{self.name}' Model."
        )

    def remove_asset(self, name: str) -> SyftSuccess | SyftError:
        asset_to_remove = None
        for asset in self.asset_list:
            if asset.name == name:
                asset_to_remove = asset
                break

        if asset_to_remove is None:
            return SyftError(message=f"No asset exists with name: {name}")
        self.asset_list.remove(asset_to_remove)
        return SyftSuccess(
            message=f"Asset '{self.name}' removed from '{self.name}' Model."
        )

    def check(self) -> Result[SyftSuccess, list[SyftError]]:
        errors = []
        for asset in self.asset_list:
            result = asset.check()
            if not result:
                errors.append(result)
        if len(errors):
            return Err(errors)
        return Ok(SyftSuccess(message="Model is Valid"))


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

@transform(CreateModelAsset, ModelAsset)
def createmodelasset_to_asset() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        add_default_node_uid
    ]

def convert_asset(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    assets = context.output.pop("asset_list", [])
    for idx, create_asset in enumerate(assets):
        asset_context = TransformContext.from_context(obj=create_asset, context=context)
        if isinstance(create_asset, CreateModelAsset):
            try:
                assets[idx] = create_asset.to(ModelAsset, context=asset_context)
            except Exception as e:
                raise e
        elif isinstance(create_asset, ModelAsset):
            assets[idx] = create_asset
    context.output["asset_list"] = assets

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
        convert_asset,
        add_current_date,
    ]
