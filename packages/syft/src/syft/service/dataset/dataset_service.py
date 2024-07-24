# stdlib
from collections.abc import Collection
from collections.abc import Sequence
import logging
from typing import cast

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..action.action_service import ActionService
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
from .dataset import Asset
from .dataset import CreateDataset
from .dataset import Dataset
from .dataset import DatasetPageView
from .dataset import DatasetUpdate
from .dataset_stash import DatasetStash

logger = logging.getLogger(__name__)


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


def _paginate_dataset_collection(
    datasets: Sequence[Dataset],
    page_size: int | None = 0,
    page_index: int | None = 0,
) -> DictTuple[str, Dataset] | DatasetPageView:
    slice_ = _paginate_collection(datasets, page_size=page_size, page_index=page_index)
    chunk = datasets[slice_] if slice_ is not None else datasets
    results = DictTuple(chunk, lambda dataset: dataset.name)

    return (
        results
        if slice_ is None
        else DatasetPageView(datasets=results, total=len(datasets))
    )


@instrument
@serializable(canonical_name="DatasetService", version=1)
class DatasetService(AbstractService):
    store: DocumentStore
    stash: DatasetStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DatasetStash(store=store)

    @service_method(
        path="dataset.add",
        name="add",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add(
        self, context: AuthedServiceContext, dataset: CreateDataset
    ) -> SyftSuccess | SyftError:
        """Add a Dataset"""
        dataset = dataset.to(Dataset, context=context)
        result = self.stash.set(
            context.credentials,
            dataset,
            add_permissions=[
                ActionObjectPermission(
                    uid=dataset.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(
            message=f"Dataset uploaded to '{context.server.name}'. "
            f"To see the datasets uploaded by a client on this server, use command `[your_client].datasets`"
        )

    @service_method(
        path="dataset.get_all",
        name="get_all",
        roles=GUEST_ROLE_LEVEL,
        warning=CRUDReminder(),
    )
    def get_all(
        self,
        context: AuthedServiceContext,
        page_size: int | None = 0,
        page_index: int | None = 0,
    ) -> DatasetPageView | DictTuple[str, Dataset] | SyftError:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        if not result.is_ok():
            return SyftError(message=result.err())

        datasets = result.ok()

        for dataset in datasets:
            if context.server is not None:
                dataset.server_uid = context.server.id
            if dataset.to_be_deleted:
                datasets.remove(dataset)

        return _paginate_dataset_collection(
            datasets=datasets, page_size=page_size, page_index=page_index
        )

    @service_method(path="dataset.search", name="search", roles=GUEST_ROLE_LEVEL)
    def search(
        self,
        context: AuthedServiceContext,
        name: str,
        page_size: int | None = 0,
        page_index: int | None = 0,
    ) -> DatasetPageView | SyftError:
        """Search a Dataset by name"""
        results = self.get_all(context)

        if isinstance(results, SyftError):
            return results

        filtered_results = [
            dataset
            for dataset_name, dataset in results.items()
            if name in dataset_name and not dataset.to_be_deleted
        ]

        return _paginate_dataset_collection(
            filtered_results, page_size=page_size, page_index=page_index
        )

    @service_method(path="dataset.get_by_id", name="get_by_id")
    def get_by_id(self, context: AuthedServiceContext, uid: UID) -> Dataset | SyftError:
        """Get a Dataset"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=result.err())
        dataset = result.ok()

        if context.server is not None:
            dataset.server_uid = context.server.id
        return dataset

    @service_method(path="dataset.get_by_action_id", name="get_by_action_id")
    def get_by_action_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> list[Dataset] | SyftError:
        """Get Datasets by an Action ID"""
        result = self.stash.search_action_ids(context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=result.err())
        datasets = result.ok()

        for dataset in datasets:
            if context.server is not None:
                dataset.server_uid = context.server.id

        return datasets

    @service_method(
        path="dataset.get_assets_by_action_id",
        name="get_assets_by_action_id",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_assets_by_action_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> list[Asset] | SyftError:
        """Get Assets by an Action ID"""
        datasets = self.get_by_action_id(context=context, uid=uid)
        if isinstance(datasets, SyftError):
            return datasets
        for dataset in datasets:
            if dataset.to_be_deleted:
                datasets.remove(dataset)
        return [
            asset
            for dataset in datasets
            for asset in dataset.asset_list
            if asset.action_id == uid
        ]

    @service_method(
        path="dataset.delete",
        name="delete",
        roles=DATA_OWNER_ROLE_LEVEL,
        warning=HighSideCRUDWarning(confirmation=True),
    )
    def delete(
        self, context: AuthedServiceContext, uid: UID, delete_assets: bool = True
    ) -> SyftSuccess | SyftError:
        """
        Soft delete: keep the dataset object, only remove the blob store entries
        After soft deleting a dataset, the user will not be able to
        see it using the `datasets.get_all` endpoint.
        Delete unique `dataset.name` key and leave UID, just rename it in case the
        user upload a new dataset with the same name.
        """
        # check if the dataset exists
        dataset = self.get_by_id(context=context, uid=uid)
        if isinstance(dataset, SyftError):
            return dataset

        return_msg = []
        if delete_assets:
            # delete the dataset's assets
            for asset in dataset.asset_list:
                msg = (
                    f"ActionObject {asset.action_id} "
                    f"linked with Assset {asset.id} "
                    f"in Dataset {uid}"
                )

                action_service = cast(
                    ActionService, context.server.get_service(ActionService)
                )
                del_res: SyftSuccess | SyftError = action_service.delete(
                    context=context, uid=asset.action_id, soft_delete=True
                )

                if isinstance(del_res, SyftError):
                    del_msg = f"Failed to delete {msg}: {del_res.message}"
                    logger.error(del_msg)
                    return del_res

                logger.info(f"Successfully deleted {msg}: {del_res.message}")

                return_msg.append(f"Asset with id '{asset.id}' successfully deleted.")

        # soft delete the dataset object from the store
        dataset_update = DatasetUpdate(
            id=uid, name=f"_deleted_{dataset.name}_{uid}", to_be_deleted=True
        )
        result = self.stash.update(context.credentials, dataset_update)
        if result.is_err():
            return SyftError(message=result.err())
        return_msg.append(f"Dataset with id '{uid}' successfully deleted.")
        return SyftSuccess(message="\n".join(return_msg))


TYPE_TO_SERVICE[Dataset] = DatasetService
SERVICE_TO_TYPES[DatasetService].update({Dataset})
