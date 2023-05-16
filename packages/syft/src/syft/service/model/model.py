from typing import Optional
from typing import List
from typing import Any
from enum import Enum
from typing import Union
from typing import Callable

from ...serde.serializable import serializable
from ...types.syft_object import SyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.uid import UID
from ...service.dataset.dataset import Contributor
from ..response import SyftError
from ..response import SyftException
from ..response import SyftSuccess
from ...service.action.model import ModelObject
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.transforms import add_current_date
from ...types.transforms import validate_url

@serializable()
class Model(SyftObject):
    # version
    __canonical_name__ = "Model"
    __version__ = SYFT_OBJECT_VERSION_1

    action_id: UID
    node_uid: UID
    name: str
    description: Optional[str]
    contributors: List[Contributor] = []
    
    def _repr_markdown_(self) -> str:
        _repr_str = f"Model: {self.name}\n"
        _repr_str += f"Pointer Id: {self.action_id}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Contributors: {len(self.contributors)}\n"
        for contributor in self.contributors:
            _repr_str += f"\t{contributor.name}: {contributor.email}\n"
        return "```python\n" + _repr_str + "\n```"
    
    @property
    def model(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(node_uid=self.node_uid)
        return api.services.action.get(self.action_id)

@serializable()
class CreateModel(SyftObject):
    # version
    __canonical_name__ = "CreateModel"
    __version__ = SYFT_OBJECT_VERSION_1
    
    id: Optional[UID] = None
    name: str
    description: Optional[str]
    contributors: List[Contributor] = []
    node_uid: Optional[UID]
    action_id: Optional[UID]
    model: Optional[ModelObject]
    
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
        
    def set_model(self, model: Any) -> None:
        if isinstance(model, SyftError):
            raise SyftException(model)
        if not isinstance(model, ModelObject):
            model = ModelObject(syft_action_data=model)
        self.model = model


class ModelUpdate:
    pass

@serializable()
class ModelInterface(SyftObject):
    __canonical_name__ = "ModelInterface"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_uid: Optional[UID]
    model_list: List[Model] # we could replace this with TwinObjects or Pointers
    current_index: int # naive way to have a versioning system
    contributors: List[Contributor] = []
    citation: Optional[str]
    url: Optional[str]
    description: Optional[str]
    updated_at: Optional[str]
    
    __attr_searchable__ = ["name", 'citation', 'url', 'description', 'action_ids']
    
    def get_current_model(self):
        return self.model_list[self.current_index]
    
    def action_ids(self) -> List[UID]:
        models = []
        for model in self.model_list:
            if model.action_id:
                models.append(model.action_id)
        return models
    
    def _repr_markdown_(self) -> str:
        _repr_str = f"Syft Model Interface: {self.name}\n"
        _repr_str += "Models:\n"
        for model in self.model_list:
            # _repr_str += f"\t{model.name}: {model.description}\n"
            _repr_str += f"\t{model.name}: {model.description}\n"

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
    
@serializable
class CreateModelInterface(ModelInterface):
    __canonical_name__ = "CreateModelInterface"
    __version__ = SYFT_OBJECT_VERSION_1
    
    id: Optional[UID] = None
    
    def set_description(self, description: str) -> None:
        self.description = description
        
    def add_citation(self, citation: str) -> None:
        self.citation = citation

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

    def add_model(self, model: CreateModel) -> None:
        self.model_list.append(model)

    def remove_model(self, name: str) -> None:
        model_to_remove = None
        for model in self.model_list:
            if model.name == name:
                model_to_remove = model
                break
            
        if model_to_remove is None:
            print(f"No model exists with name: {name}")    
        self.mode_list.remove(model_to_remove)

class ModelInterfaceUpdate:
    pass

@transform(CreateModel, Model)
def createmodel_to_model() -> List[Callable]:
    return [generate_id]

@transform(CreateModelInterface, ModelInterface)
def createMI_to_MI() -> List[Callable]:
    return [generate_id, validate_url, add_current_date("update_at")]
