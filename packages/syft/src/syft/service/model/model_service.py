# stdlib
from collections.abc import Collection
from collections.abc import Sequence

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..warnings import CRUDReminder
from ..warnings import HighSideCRUDWarning
from .model import CreateModel
from .model import CreateModelAsset
from .model import Model
from .model import ModelAsset
from .model import ModelPageView
from .model_stash import ModelStash


def _paginate_collection(
    collection: Collection,
    page_size: int | None = 0,
    page_index: int | None = 0,
) -> slice | None:
    if page_size is None or page_size <= 0:
        return None

    # If chunk size is defined, then split list into evenly sized chunks
    total = len(collection)
    page_index = 0 if page_index is None else page_index

    if page_size > total or page_index >= total // page_size or page_index < 0:
        return None

    start = page_size * page_index
    stop = min(page_size * (page_index + 1), total)
    return slice(start, stop)


def _paginate_model_collection(
    models: Sequence[Model],
    page_size: int | None = 0,
    page_index: int | None = 0,
) -> DictTuple[str, Model] | ModelPageView:
    slice_ = _paginate_collection(models, page_size=page_size, page_index=page_index)
    chunk = models[slice_] if slice_ is not None else models
    results = DictTuple(chunk, lambda model: model.name)

    return (
        results if slice_ is None else ModelPageView(models=results, total=len(models))
    )


@instrument
@serializable()
class ModelService(AbstractService):
    store: DocumentStore
    stash: ModelStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ModelStash(store=store)

    @service_method(
        path="model.add",
        name="add",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add(
        self, context: AuthedServiceContext, model: CreateModel
    ) -> SyftSuccess | SyftError:
        """Add a model"""
        model = model.to(Model, context=context)

        print("got model", model)

        result = self.stash.set(
            context.credentials,
            model,
            add_permissions=[
                ActionObjectPermission(
                    uid=model.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(
            message=f"Model uploaded to '{context.node.name}'. "
            f"To see the models uploaded by a client on this node, use command `[your_client].models`"
        )

    @service_method(
        path="model.get_all",
        name="get_all",
        roles=GUEST_ROLE_LEVEL,
        warning=CRUDReminder(),
    )
    def get_all(
        self,
        context: AuthedServiceContext,
        page_size: int | None = 0,
        page_index: int | None = 0,
    ) -> ModelPageView | DictTuple[str, Model] | SyftError:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        if not result.is_ok():
            return SyftError(message=result.err())

        models = result.ok()

        return _paginate_model_collection(
            models=models, page_size=page_size, page_index=page_index
        )

    @service_method(
        path="model.delete_by_uid",
        name="delete_by_uid",
        roles=DATA_OWNER_ROLE_LEVEL,
        warning=HighSideCRUDWarning(confirmation=True),
    )
    def delete_model(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())


TYPE_TO_SERVICE[Model] = ModelService
SERVICE_TO_TYPES[ModelService].update({Model})
