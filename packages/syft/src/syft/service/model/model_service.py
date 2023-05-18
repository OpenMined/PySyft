
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
from .model import Model
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
        self.stash = ModelInterfaceStash(store=store)
    
    @service_method(path="model_interface.add", name="add", roles=DATA_OWNER_ROLE_LEVEL)
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
    
    @service_method(path="model_interface.get_by_id", name="get_by_id")
    def get_by_id(
        self, context: AuthedServiceContext, uid:UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get Model Interface by id"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            model_interface = result.ok()
            model_interface.node_uid = context.node.id
            return model_interface
        return SyftError(message=result.err())
    
    @service_method(path="model_interface.get_by_action_id", name="get_by_action_id")
    def get_by_action_id(
        self, context: AuthedServiceContext, uid:UID
    ) -> Union[List[ModelInterface], SyftError]:
        """"""
        result = self.stash.search_action_ids(context.credentials, uid=uid)
        if result.is_ok():
            model_interfacess = result.ok()
            for model_interfaces in model_interfacess:
                model_interfaces.node_uid = model_interfaces.node.id
            return model_interfacess
        return SyftError(message=result.err())
    
    @service_method(
        path="model_interface.get_models_by_action_id", name="get_models_by_action_id"
    )
    def get_models_by_action_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[List[Model], SyftError]:
        model_interfaces = self.get_by_action_id(context=context, uid=uid)
        models = []
        if issubclass(model_interfaces, list):
            for model_interface in model_interfaces:
                for model in model_interface.model_list:
                    models.append(model)
        elif isinstance(model_interfaces, SyftError):
            return model_interfaces
        return models

    @service_method(path="model_interface.search", name="search")
    def search(
        self, context: AuthedServiceContext, name: str 
    )-> Union[List[ModelInterface], SyftError]:
        """Search a Model Interface by name"""
        results = self.get_all(context)

        return (
            results
            if isinstance(results, SyftError)
            else [model for model in results if name in model.name]
        )

    def delete_model_interface():
        pass
    
    def update_model_interface():
        pass
    
    def update_model():
        pass


TYPE_TO_SERVICE[ModelInterface] = ModelInterfaceService
SERVICE_TO_TYPES[ModelInterfaceService].update({ModelInterface})

    