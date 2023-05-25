
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
from .model_stash import ModelCardStash
from ..context import AuthedServiceContext
from .model import CreateModelCard
from .model import ModelCard
from ..response import SyftError
from ..response import SyftSuccess
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ...types.syft_file import CreateSyftFile

@instrument
@serializable()
class ModelCardService(AbstractService):
    store: DocumentStore
    stash: ModelCardStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ModelCardStash(store=store)
    
    @service_method(path="model_card.add_from_files", name="add_from_files", roles=DATA_OWNER_ROLE_LEVEL)
    def add_from_files(
        self, context: AuthedServiceContext, files: List[CreateSyftFile], name: str
    ) -> Union[SyftSuccess, SyftError]:
        
        model_card = ModelCard(
            name=name
        )
        
        result = self.stash.set(
            context.credentials,
            model_card,
            add_permissions=[
                ActionObjectPermission(
                    uid=model_card.id, permission=ActionPermission.ALL_READ
                ),
            ]
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Model Card Added")
    
    @service_method(
        path="model_card.add_from_safetensors", 
        name="add_from_safetensors", 
        roles=DATA_OWNER_ROLE_LEVEL
    )
    def add_from_safetensors(
        self, context: AuthedServiceContext, files: List[CreateSyftFile], name: str
    ) -> Union[SyftSuccess, SyftError]:
        pass
    
    
    @service_method(path="model_card.add", name="add", roles=DATA_OWNER_ROLE_LEVEL)
    def add(
        self, context: AuthedServiceContext, model_card: CreateModelCard
    ) -> Union[SyftSuccess, SyftError]:
        """Add a ModelCard"""
        model_card = model_card.to(ModelCard, context=context)
        result = self.stash.set(
            context.credentials,
            model_card,
            add_permissions=[
                ActionObjectPermission(
                    uid=model_card.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Model Interface Added")
    
    @service_method(path="model_card.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> Union[List[ModelCard], SyftError]:
        """Get all Model Interfaces"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            model_cards = result.ok()
            results = []
            for model_card in model_cards:
                model_card.node_uid = context.node.id
                results.append(model_card)
            return results
        return SyftError(message=result.err())
    
    @service_method(path="model_card.get_by_id", name="get_by_id")
    def get_by_id(
        self, context: AuthedServiceContext, uid:UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get Model Interface by id"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            model_card = result.ok()
            model_card.node_uid = context.node.id
            return model_card
        return SyftError(message=result.err())
    
    @service_method(path="model_card.get_by_action_id", name="get_by_action_id")
    def get_by_action_id(
        self, context: AuthedServiceContext, uid:UID
    ) -> Union[List[ModelCard], SyftError]:
        """"""
        result = self.stash.search_action_ids(context.credentials, uid=uid)
        if result.is_ok():
            model_cardss = result.ok()
            for model_cards in model_cardss:
                model_cards.node_uid = model_cards.node.id
            return model_cardss
        return SyftError(message=result.err())
    
    @service_method(
        path="model_card.get_models_by_action_id", name="get_models_by_action_id"
    )
    # def get_models_by_action_id(
    #     self, context: AuthedServiceContext, uid: UID
    # ) -> Union[List[Model], SyftError]:
    #     model_cards = self.get_by_action_id(context=context, uid=uid)
    #     models = []
    #     if issubclass(model_cards, list):
    #         for model_card in model_cards:
    #             for model in model_card.model_list:
    #                 models.append(model)
    #     elif isinstance(model_cards, SyftError):
    #         return model_cards
    #     return models

    @service_method(path="model_card.search", name="search")
    def search(
        self, context: AuthedServiceContext, name: str 
    )-> Union[List[ModelCard], SyftError]:
        """Search a Model Interface by name"""
        results = self.get_all(context)

        return (
            results
            if isinstance(results, SyftError)
            else [model for model in results if name in model.name]
        )

    def delete_model_card():
        pass
    
    def update_model_card():
        pass
    
    def update_model():
        pass


TYPE_TO_SERVICE[ModelCard] = ModelCardService
SERVICE_TO_TYPES[ModelCardService].update({ModelCard})

    