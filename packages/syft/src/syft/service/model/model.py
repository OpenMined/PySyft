# stdlib
from collections.abc import Callable
from datetime import datetime
from enum import Enum
import hashlib
import os
import random
from string import Template
from textwrap import dedent
from typing import Any
from typing import ClassVar
from typing import cast

# third party
from IPython.display import HTML
from IPython.display import Markdown
from IPython.display import display
from pydantic import ConfigDict
from result import Err
from result import Ok
from result import OkErr
from result import Result
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# relative
from ...client.api import APIRegistry
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...serde.serialize import _serialize as serialize
from ...types.datetime import DateTime
from ...types.dicttuple import DictTuple
from ...types.file import SyftFolder
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.transforms import validate_url
from ...types.uid import UID
from ...util.markdown import as_markdown_python_code
from ...util.notebook_ui.components.sync import CopyIDButton
from ..action.action_object import ActionDataEmpty
from ..action.action_object import ActionObject
from ..action.action_object import BASE_PASSTHROUGH_ATTRS
from ..action.action_service import ActionService
from ..context import AuthedServiceContext
from ..dataset.dataset import Contributor
from ..dataset.dataset import MarkdownDescription
from ..policy.policy import get_code_from_class
from ..response import SyftError
from ..response import SyftSuccess
from ..response import SyftWarning
from .model_html_template import asset_repr_template
from .model_html_template import generate_attr_html
from .model_html_template import model_repr_template


def has_permission(data_result: Any) -> bool:
    # TODO: implement in a better way
    return not (
        isinstance(data_result, str)
        and data_result.startswith("Permission")
        and data_result.endswith("denied")
    )


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
    server_uid: UID
    created_at: DateTime = DateTime.now()
    asset_hash: str

    __repr_attrs__ = ["name", "created_at", "asset_hash"]

    def __init__(
        self,
        description: MarkdownDescription | str | None = "",
        **kwargs: Any,
    ):
        if isinstance(description, str):
            description = MarkdownDescription(text=description)
        super().__init__(**kwargs, description=description)

    def _ipython_display_(self) -> None:
        if self.description:
            string = f"""<details>
        <summary>Show Asset Description:</summary>
        {self.description._repr_markdown_()}
    </details>"""
            display(HTML(self._repr_html_()), Markdown(string))
        else:
            display(HTML(self._repr_html_()))

    def _repr_html_(self) -> Any:
        identifier = random.randint(1, 2**32)  # nosec
        result_tab_id = f"Result_{identifier}"
        logs_tab_id = f"Logs_{identifier}"
        model_object_type = "Asset"
        api_header = "model_assets/"
        model_name = f"{self.name}"
        button_html = CopyIDButton(copy_text=str(self.id), max_width=60).to_html()

        attrs = {
            "Created at": str(self.created_at),
            "Action ID": str(self.action_id),
            "Server ID": str(self.server_uid),
            "Asset Hash": str(self.asset_hash),
        }
        attrs_html = generate_attr_html(attrs)

        template = Template(asset_repr_template)
        return template.substitute(
            model_object_type=model_object_type,
            api_header=api_header,
            model_name=model_name,
            button_html=button_html,
            attrs_html=attrs_html,
            identifier=identifier,
            result_tab_id=result_tab_id,
            logs_tab_id=logs_tab_id,
            result=None,
        )

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

    @property
    def data(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            server_uid=self.server_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None or api.services is None:
            return None
        res = api.services.action.get(self.action_id)
        if has_permission(res):
            return res.syft_action_data
        else:
            warning = SyftWarning(
                message="You do not have permission to access private data."
            )
            display(warning)
            return None

    @property
    def mock(self) -> SyftError | Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(f"api is None. You must login to {self.syft_server_uid}")
        result = api.services.action.get_mock(self.action_id)
        if isinstance(result, SyftError):
            return result
        try:
            if isinstance(result, SyftObject):
                return result.syft_action_data
            return result
        except Exception as e:
            return SyftError(message=f"Failed to get mock. {e}")

    # def __call__(self, *args, **kwargs) -> Any:
    #     endpoint = self.endpoint
    #     result = endpoint.__call__(*args, **kwargs)
    #     return result

@serializable()
class SubmitModelCode(ActionObject):
    # version
    __canonical_name__ = "SubmitModelCode"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[type] = str
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS + [
        "code",
        "class_name",
        "__call__",
    ]

    class_name: str

    @property
    def code(self) -> str:
        return self.syft_action_data

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        return as_markdown_python_code(self.code)

    def __call__(self, **kwargs: dict) -> Any:
        # Load Class
        exec(self.code)

        # execute it
        func_string = f"{self.class_name}(**kwargs)"
        result = eval(func_string, None, locals())  # nosec

        return result

    __repr_attrs__ = ["class_name", "code"]


@serializable(canonical_name="SyftModelClass", version=1)
class SyftModelClass:
    def __init__(self, assets: list[ModelAsset]) -> None:
        self.__user_init__(assets)

    def __user_init__(self, assets: list[ModelAsset]) -> None:
        pass

    def inference(self) -> Any:
        pass
    
    def generate_mock_assets(self) -> Any:
        pass

@serializable(canonical_name="HFModelClass", version=1)
class HFModelClass(SyftModelClass):
    repo_id: str = None
    
    def __user_init__(self, assets: list) -> None:
        model_folder = assets[0]
        model_folder = str(model_folder.model_folder)
        print(model_folder, type(model_folder))
        self.model = AutoModelForCausalLM.from_pretrained(model_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder)

    def __call__(self, prompt: str, raw=False, **kwargs) -> str:
        # Makes the model callable for direct predictions.
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
            **kwargs,
        )
        if raw:
            return gen_tokens
        else:
            gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
            return gen_text

    def inference(self, prompt: str, raw=False, **kwargs) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
            **kwargs,
        )
        if raw:
            return gen_tokens
        else:
            gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
            return gen_text

    def inference_dump(self, prompt: str):
        encoded_input = self.tokenizer(prompt, return_tensors="pt")
        return self.model(**encoded_input)
    
    def generate_mock_assets(self):
        mock_model = AutoModelForCausalLM.from_config(self.model.config_class())
        mock_model.save_pretrained("/tmp/mock_weights")
        self.tokenizer.save_pretrained("/tmp/mock_weights")
        mock_folder = SyftFolder.from_dir(name="mock", path="/tmp/mock_weights")
        return mock_folder
        

    # Exposes the HF well-known API
    def tokenize(text):
        # Tokenizes a given text.
        pass

    def decode(token_ids):
        # Converts token IDs back to text.
        pass

    def train():
        # Puts the model in training mode.
        pass

    def eval():
        # Puts the model in evaluation mode.
        pass

    def forward(input_ids, attention_mask, labels=None):
        # Defines the forward pass for the model.
        pass

# @syft_model(name="gpt2")
# class GPT2ModelClass(HFModelClass):
#     repo_id = "openai-community/gpt2"
    
def syft_model(
    name: str | None = None,
) -> Callable:
    def decorator(cls: Any) -> Callable:
        try:
            code = dedent(get_code_from_class(cls))
            code = f"import syft as sy\n{code}"
            class_name = cls.__name__
            res = SubmitModelCode(syft_action_data_cache=code, class_name=class_name)
        except Exception as e:
            raise e

        success_message = SyftSuccess(
            message=f"Syft Model Class '{cls.__name__}' successfully created. "
        )
        display(success_message)
        return res

    return decorator

@serializable()
class CreateModelAsset(SyftObject):
    # version
    __canonical_name__ = "CreateModelAsset"
    __version__ = SYFT_OBJECT_VERSION_1

    __repr_attrs__ = ["name", "description", "contributors", "data", "created_at"]

    name: str
    server_uid: UID | None = None
    description: MarkdownDescription | None = None
    contributors: set[Contributor] = set()
    data: Any | None = None  # SyftFolder will go here!
    mock: Any | None = None
    created_at: DateTime | None = None
    action_id: UID | None = None

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, description: str | None = "", **kwargs: Any) -> None:
        if 'data' in kwargs:
            if isinstance(kwargs['data'], str) and os.path.exists(os.path.dirname(kwargs['data'])):
                model_folder = SyftFolder.from_dir(name=kwargs['name']+"_data", path=kwargs['data'])
                kwargs['data'] = model_folder

        if 'mock' in kwargs:
            if isinstance(kwargs['mock'], str) and os.path.exists(os.path.dirname(kwargs['mock'])):
                model_folder = SyftFolder.from_dir(name=kwargs['name']+"_mock", path=kwargs['mock'])
                kwargs['mock'] = model_folder

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

    def _ipython_display_(self) -> None:
        display(HTML(self._repr_html_()))
        if self.description:
            string = f"""<details>
        <summary>Show Asset Description:</summary>
        {self.description._repr_markdown_()}
    </details>"""
            display(Markdown(string))

    def _repr_html_(self) -> Any:
        identifier = random.randint(1, 2**32)  # nosec
        result_tab_id = f"Result_{identifier}"
        logs_tab_id = f"Logs_{identifier}"
        model_object_type = "Asset"
        api_header = "model_assets/"
        model_name = f"{self.name}"
        button_html = CopyIDButton(copy_text=str(self.id), max_width=60).to_html()

        attrs = {
            "Created at": str(self.created_at),
        }
        attrs_html = generate_attr_html(attrs)

        template = Template(asset_repr_template)
        return template.substitute(
            model_object_type=model_object_type,
            api_header=api_header,
            model_name=model_name,
            button_html=button_html,
            attrs_html=attrs_html,
            identifier=identifier,
            result_tab_id=result_tab_id,
            logs_tab_id=logs_tab_id,
            result=None,
        )


@serializable()
class Model(SyftObject):
    # version
    __canonical_name__: str = "Model"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["name", "citation", "url", "card"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "url", "created_at"]

    name: str
    asset_list: list[ModelAsset] = []
    server_uid: UID
    contributors: set[Contributor] = set()
    citation: str | None = None
    url: str | None = None
    card: MarkdownDescription | None = None
    updated_at: str | None = None
    created_at: DateTime = DateTime.now()
    show_code: bool = False
    show_interface: bool = True
    example_text: str | None = None
    mb_size: float | None = None
    code_action_id: UID | None = None
    syft_model_hash: str | None = None
    autogenerate_mock: bool = False

    @property
    def server_name(self) -> str | SyftError | None:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            # return "SyftError(
            #     message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            # )"
            return "unknown"
        return api.server_name

    def __init__(
        self,
        card: str | MarkdownDescription | None = "",
        **kwargs: Any,
    ) -> None:
        if isinstance(card, str):
            card = MarkdownDescription(text=card)
        super().__init__(**kwargs, card=card)

    @property
    def icon(self) -> str:
        return "no icon"

    @property
    def model_code(self) -> SubmitModelCode | None:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None or api.services is None:
            return None
        res = api.services.action.get_model_code(self.code_action_id)
        if has_permission(res):
            return res
        else:
            warning = SyftWarning(
                message="You do not have permission to access private data."
            )
            display(warning)
            return None

    @property
    def mock(self) -> SyftModelClass:
        model_code = self.model_code
        if model_code is None:
            raise ValueError("[Model.mock] Cannot access model code")
        mock_assets = [asset.mock for asset in self.asset_list]
        return model_code(assets=mock_assets)

    @property
    def data(self) -> SyftModelClass:
        model_code = self.model_code
        if model_code is None:
            raise ValueError("[Model.mock] Cannot access model code")
        data_assets = [asset.data for asset in self.asset_list]
        return model_code(assets=data_assets)

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Assets": len(self.asset_list),
            "Url": self.url,
            "Size": f"{self.mb_size:.2f} (MB)" if self.mb_size else "Unknown",
            "created at": str(self.created_at),
        }

    def _ipython_display_(self) -> None:
        show_string = "For more information, `.assets` reveals the resources \
            used by the model and `.model_code` will show the model code."
        if self.card:
            card_string = f"""<details>
        <summary>Show model card:</summary>
        {self.card._repr_markdown_()}
    </details>"""
            display(
                HTML(self._repr_html_()),
                Markdown(card_string),
                Markdown(show_string),
            )
        else:
            display(HTML(self._repr_html_()), Markdown(show_string))

    def _repr_html_(self) -> Any:
        # TODO: Improve Repr
        # return f"Model Hash: {self.syft_model_hash}"
        identifier = random.randint(1, 2**32)  # nosec
        result_tab_id = f"Result_{identifier}"
        logs_tab_id = f"Logs_{identifier}"
        model_object_type = "Model"
        api_header = f"{self.server_name}/models/"
        model_name = f"{self.name}"
        button_html = CopyIDButton(copy_text=str(self.id), max_width=60).to_html()

        attrs = {
            "Size": f"{self.mb_size:.2f} (MB)" if self.mb_size else "Unknown",
            "URL": str(self.url),
            "Created at": str(self.created_at),
            "Updated at": self.updated_at,
            "Citation": self.citation,
            "Model Hash": self.syft_model_hash,
        }
        attrs_html = generate_attr_html(attrs)
        template = Template(model_repr_template)
        return template.substitute(
            model_object_type=model_object_type,
            api_header=api_header,
            model_name=model_name,
            button_html=button_html,
            attrs_html=attrs_html,
            identifier=identifier,
            result_tab_id=result_tab_id,
            logs_tab_id=logs_tab_id,
        )

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
        if self.card:
            _repr_str += f"card:\n{self.card.text}\n"
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
        if self.card:
            _repr_str += f"card: \n\n{self.card.text}\n\n"
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

    code: SubmitModelCode
    code_action_id: UID | None = None
    asset_list: list[Any] = []
    created_at: DateTime | None = None  # type: ignore[assignment]
    model_config = ConfigDict(validate_assignment=True)
    server_uid: UID | None = None  # type: ignore[assignment]

    def __init__(
        self,
        code: type | SubmitModelCode,
        **kwargs: Any,
    ) -> None:
        if isinstance(code, type) and issubclass(code, SyftModelClass):
            code = syft_model(name='test')(code)
        super().__init__(**kwargs, code=code)

    def set_card(self, card: str) -> None:
        self.card = MarkdownDescription(text=card)

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


def add_default_server_uid(context: TransformContext) -> TransformContext:
    if context.output is not None:
        if context.output["server_uid"] is None and context.server is not None:
            context.output["server_uid"] = context.server.id
    else:
        raise ValueError(f"{context}'s output is None. No transformation happened")
    return context


def add_asset_hash(context: TransformContext) -> TransformContext:
    # relative

    if context.output is None:
        return context

    if context.server is None:
        raise ValueError("Context should have a server attached to it.")

    action_id = context.output["action_id"]
    if action_id is not None:
        action_service = context.server.get_service(ActionService)
        # Q: Why is service returning an result object [Ok, Err]?
        action_obj = action_service.get(context=context, uid=action_id)

        if action_obj.is_err():
            return SyftError(f"Failed to get action object with id {action_obj.err()}")
        # NOTE: for a TwinObject, this hash of the private data
        context.output["asset_hash"] = action_obj.ok().hash()
    else:
        raise ValueError("Model Asset must have an action_id to generate a hash")

    return context


@transform(CreateModelAsset, ModelAsset)
def createmodelasset_to_asset() -> list[Callable]:
    return [generate_id, add_msg_creation_time, add_default_server_uid, add_asset_hash]


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


def add_model_hash(context: TransformContext) -> TransformContext:
    # relative

    if context.output is None:
        return context

    if context.server is None:
        raise ValueError("Context should have a server attached to it.")

    self_id = context.output["id"]
    if self_id is not None:
        action_service = context.server.get_service(ActionService)
        # Q: Why is service returning an result object [Ok, Err]?
        model_ref_action_obj = action_service.get(context=context, uid=self_id)

        if model_ref_action_obj.is_err():
            return SyftError(
                f"[Model]Failed to get action object with id {model_ref_action_obj.err()}"
            )
        context.output["syft_model_hash"] = model_ref_action_obj.ok().hash(
            context=context
        )
    else:
        raise ValueError("Model  must have an valid ID")

    return context


def add_server_uid(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context
    if context.server:
        context.output["server_uid"] = context.server.id
    return context


@transform(CreateModel, Model)
def createmodel_to_model() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        validate_url,
        # generate_mock,
        convert_asset,
        add_current_date,
        add_model_hash,
        add_server_uid,
    ]


@serializable()
class ModelRef(ActionObject):
    __canonical_name__ = "ModelRef"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[type] = list[UID]
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS + [
        "ref_objs",
        "load_model",
        "load_data",
        "store_ref_objs_to_store",
    ]
    ref_objs: list = []  # Contains the loaded data

    # Schema:
    # [model_code_id, asset1_id, asset2_id, ...]

    def store_ref_objs_to_store(
        self, context: AuthedServiceContext, clear_ref_objs: bool = False
    ) -> SyftError | None:
        admin_client = context.server.root_client

        if not self.ref_objs:
            return SyftError(message="No ref_objs to store in Model Ref")

        for ref_obj in self.ref_objs:
            res = admin_client.services.action.set(ref_obj)
            if isinstance(res, SyftError):
                return res

        if clear_ref_objs:
            self.ref_objs = []

        model_ref_res = admin_client.services.action.set(self)
        if isinstance(model_ref_res, SyftError):
            return model_ref_res

        return None

    def hash(
        self,
        recalculate: bool = False,
        context: TransformContext | None = None,
        client: SyftClient | None = None,
    ) -> str:
        if context is None and client is None:
            raise ValueError(
                "Either context or client should be provided to ModelRef.hash()"
            )
        if context and context.server is None:
            raise ValueError("Context should have a server attached to it.")

        self.syft_action_data_hash: str | None
        if not recalculate and self.syft_action_data_hash:
            return self.syft_action_data_hash

        if not self.ref_objs:
            if context:
                action_objs = self.load_data(context)
            else:
                action_objs = self.load_data(self_client=client)
        else:
            action_objs = self.ref_objs

        hash_items = [action_obj.hash() for action_obj in action_objs]
        hash_bytes = serialize(hash_items, to_bytes=True)
        hash_str = hashlib.sha256(hash_bytes).hexdigest()
        self.syft_action_data_hash = hash_str
        return self.syft_action_data_hash

    def load_data(
        self,
        context: AuthedServiceContext | None = None,
        self_client: SyftClient | None = None,
        wrap_ref_to_obj: bool = False,
        unwrap_action_data: bool = True,
        remote_client: SyftClient | None = None,
    ) -> list:
        if context is None and self_client is None:
            raise ValueError(
                "Either context or client should be provided to ModelRef.load_data()"
            )

        client = context.server.root_client if context else self_client

        code_action_id = self.syft_action_data[0]
        asset_action_ids = self.syft_action_data[1::]

        model = client.api.services.action.get(code_action_id)

        if isinstance(model, OkErr):
            if model.is_err():
                return SyftError(message=f"Failed to load model code:{model.err()}")
            model = model.ok()

        asset_list = []
        for asset_action_id in asset_action_ids:
            action_object = client.api.services.action.get(asset_action_id)
            if isinstance(action_object, OkErr):
                if action_object.is_err():
                    return SyftError(
                        message=f"Failed to load asset:{action_object.err()}"
                    )
                action_object = action_object.ok()
            action_data = action_object.syft_action_data

            # Save to blob storage of remote client if provided
            if remote_client is not None:
                action_object.syft_blob_storage_entry_id = None
                blob_res = action_object._save_to_blob_storage(client=remote_client)
                # For smaller data, we do not store in blob storage
                # so for the cases, where we store in blob storage
                # we need to clear the cache , to avoid sending the data again
                # stdlib

                action_object.syft_blob_storage_entry_id = cast(
                    UID | None, action_object.syft_blob_storage_entry_id
                )
                if action_object.syft_blob_storage_entry_id:
                    action_object._clear_cache()
                if isinstance(blob_res, SyftError):
                    return blob_res
                # TODO: fix Tech Debt
                # Currently, Setting the Location of the object to the remote client
                # As this is later used by the enclave to fetch the syft_action_data
                # in reload_cache method of action object
                # This is a quick fix to address the same
                action_object._set_obj_location_(
                    remote_client.id, context.server.signing_key.verify_key
                )
            asset_list.append(action_data if unwrap_action_data else action_object)

        loaded_data = [model] + asset_list
        if wrap_ref_to_obj:
            self.ref_objs = loaded_data

        return loaded_data

    def load_model(self, context: AuthedServiceContext) -> SyftModelClass:
        loaded_data = self.load_data(context)
        model = loaded_data[0]
        asset_list = loaded_data[1::]

        loaded_model = model(assets=asset_list)
        return loaded_model
