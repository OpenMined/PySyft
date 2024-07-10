# stdlib
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from textwrap import dedent
from typing import Any
from typing import ClassVar

# third party
from IPython.display import display
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
from ...util.markdown import as_markdown_python_code
from ..action.action_object import ActionDataEmpty
from ..action.action_object import ActionObject
from ..action.action_object import BASE_PASSTHROUGH_ATTRS
from ..dataset.dataset import Contributor
from ..dataset.dataset import MarkdownDescription
from ..policy.policy import get_code_from_class
from ..response import SyftError
from ..response import SyftSuccess
from ..response import SyftWarning


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

    __repr_attrs__ = ["name", "url"]

    name: str
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    action_id: UID
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
        return "todo a table"

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
            and self.action_id == other.action_id
            and self.created_at == other.created_at
        )

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
        if api is None or api.services is None:
            return None
        res = api.services.action.get(self.action_id)
        if self.has_permission(res):
            return res.syft_action_data
        else:
            warning = SyftWarning(
                message="You do not have permission to access private data."
            )
            display(warning)
            return None

    # def __call__(self, *args, **kwargs) -> Any:
    #     endpoint = self.endpoint
    #     result = endpoint.__call__(*args, **kwargs)
    #     return result


@serializable()
class SyftModelClass:
    def __init__(self, assets: list[ModelAsset]) -> None:
        self.__user_init__(assets)

    def __user_init__(self, assets: list[ModelAsset]) -> None:
        pass

    def inference(self) -> Any:
        pass


def syft_model(
    name: str | None = None,
) -> Callable:
    def decorator(cls: Any) -> Callable:
        try:
            code = dedent(get_code_from_class(cls))
            code = f"import syft as sy\n{code}"
            class_name = cls.__name__
            res = SubmitModelCode(code=code, class_name=class_name)
        except Exception as e:
            print("e", e)
            raise e

        success_message = SyftSuccess(
            message=f"Syft Model Class '{cls.__name__}' successfully created. "
        )
        display(success_message)
        return res

    return decorator


@serializable()
class SubmitModelCode(SyftObject):
    # version
    __canonical_name__ = "SubmitModelCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore[assignment]
    code: str
    class_name: str
    # signature: inspect.Signature

    def __call__(self, **kwargs) -> Any:
        # Load Class
        exec(self.code)

        # execute it
        func_string = f"{self.class_name}(**kwargs)"
        result = eval(func_string, None, locals())  # nosec

        return result

    __repr_attrs__ = ["class_name", "code"]


@serializable()
class CreateModelAsset(SyftObject):
    # version
    __canonical_name__ = "CreateModelAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    __repr_attrs__ = ["name", "description", "contributors", "data", "created_at"]

    name: str
    node_uid: UID | None = None
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    data: Any | None = None  # SyftFolder will go here!
    mock: Any | None = None
    created_at: DateTime | None = None
    action_id: UID | None = None

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, description: str | None = "", **kwargs: Any) -> None:
        super().__init__(
            **kwargs, description=MarkdownDescription(text=str(description))
        )

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

    def check(self) -> SyftSuccess | SyftError:
        return SyftSuccess(message="Model Asset is Valid")

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


@serializable()
class Model(SyftObject):
    # version
    __canonical_name__: str = "Model"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["name", "citation", "url", "description"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "url", "created_at"]

    name: str
    submit_model: SubmitModelCode
    asset_list: list[ModelAsset] = []
    contributors: set[Contributor] = set()
    citation: str | None = None
    url: str | None = None
    description: MarkdownDescription | None = None
    updated_at: str | None = None
    created_at: DateTime = DateTime.now()
    show_code: bool = False
    show_interface: bool = True
    example_text: str | None = None
    mb_size: float | None = None

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
        return "no icon"

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Assets": len(self.asset_list),
            "Url": self.url,
            "Size": f"{self.mb_size} (MB)",
            "created at": str(self.created_at),
        }

    # def _repr_html_(self) -> Any:
    #     return "todo table"

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
            _repr_str += f"Description:\n{self.description.text}\n"
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
        if self.example_text:
            _repr_str += f"Example: \n\n{self.example_text}\n\n"
        return _repr_str

    # @property
    # def run(self) -> Callable | None:
    #     warning = SyftWarning(
    #         message="This code was submitted by a User and could be UNSAFE."
    #     )
    #     display(warning)

    #     # 🟡 TODO: re-use the same infrastructure as the execute_byte_code function
    #     def wrapper(*args: Any, **kwargs: Any) -> Callable | SyftError:
    #         try:
    #             filtered_kwargs = {}
    #             on_private_data, on_mock_data = False, False
    #             for k, v in kwargs.items():
    #                 filtered_kwargs[k], arg_type = debox_asset(v)
    #                 on_private_data = (
    #                     on_private_data or arg_type == ArgumentType.PRIVATE
    #                 )
    #                 on_mock_data = on_mock_data or arg_type == ArgumentType.MOCK

    #             if on_private_data:
    #                 display(
    #                     SyftInfo(
    #                         message="The result you see is computed on PRIVATE data."
    #                     )
    #                 )
    #             if on_mock_data:
    #                 display(
    #                     SyftInfo(message="The result you see is computed on MOCK data.")
    #                 )

    #             # remove the decorator
    #             inner_function = ast.parse(self.raw_code).body[0]
    #             inner_function.decorator_list = []
    #             # compile the function
    #             raw_byte_code = compile_byte_code(unparse(inner_function))
    #             # load it
    #             exec(raw_byte_code)  # nosec
    #             # execute it
    #             evil_string = f"{self.service_func_name}(**filtered_kwargs)"
    #             result = eval(evil_string, None, locals())  # nosec
    #             # return the results
    #             return result
    #         except Exception as e:
    #             return SyftError(f"Failed to execute 'run'. Error: {e}")

    #     return wrapper


@serializable()
class CreateModel(Model):
    # version
    __canonical_name__ = "CreateModel"
    __version__ = SYFT_OBJECT_VERSION_1

    __repr_attrs__ = ["name", "url"]

    submit_model: SubmitModelCode
    asset_list: list[Any] = []
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
    return [generate_id, add_msg_creation_time, add_default_node_uid]


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


@serializable()
class ModelRef(ActionObject):
    __canonical_name__ = "ModelRef"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[type] = UID
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS
