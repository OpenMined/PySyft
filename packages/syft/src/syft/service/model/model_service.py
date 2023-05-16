
from typing import List
from typing import Union

from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .model_stash import ModelInterfaceStash
from ..context import AuthedServiceContext
from .model import CreateModelInterface
from .model import ModelInterface
from ..response import SyftError
from ..response import SyftSuccess
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL


@instrument
@serializable()
class ModelInterfaceService(AbstractService):
    store: DocumentStore
    stash: ModelInterfaceStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ModelInterfaceService(store=store)
    
    @service_method(path="model_interface.add", name="add", role=DATA_OWNER_ROLE_LEVEL)
    def add(
        self, context: AuthedServiceContext, model_interface: CreateModelInterface
    ) -> Union[SyftSuccess, SyftError]:
        """Add a ModelInterface"""
        model_interface = model_interface.to(ModelInterface, context=context)
        result = self.stash.set(
            context.credentials,
            model_interface,
            add_permissions=[
                ActionObjectPermission(
                    uid=model_interface.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Model Interface Added")
    
    @service_method(path="model_interface.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> Union[List[ModelInterface], SyftError]:
        """Get all Model Interfaces"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            model_interfaces = result.ok()
            results = []
            for model_interface in model_interfaces:
                model_interface.node_uid = context.node.id
                results.append(model_interface)
            return results
        return SyftError(message=result.err())
    
    def get_by_id():
        pass
    
    def get_by_action_id():
        pass
    
    def get_models_by_action_id():
        pass

    def search():
        pass
    
    def delete_model_interface():
        pass
    
    def update_model_interface():
        pass
    
    def update_model():
        pass


TYPE_TO_SERVICE[ModelInterface] = ModelInterfaceService
SERVICE_TO_TYPES[ModelInterfaceService].update({ModelInterface})

    