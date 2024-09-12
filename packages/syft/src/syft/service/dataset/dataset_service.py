# stdlib
from collections.abc import Collection
from collections.abc import Sequence
import logging

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
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


@serializable(canonical_name="DatasetService", version=1)
class DatasetService(AbstractService):
    stash: DatasetStash

    def __init__(self, store: DBManager) -> None:
        self.stash = DatasetStash(store=store)

    @service_method(
        path="dataset.add",
        name="add",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def add(self, context: AuthedServiceContext, dataset: CreateDataset) -> SyftSuccess:
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
        ).unwrap()

        return SyftSuccess(
            message=(
                f"Dataset uploaded to '{context.server.name}'."
                f" To see the datasets uploaded by a client on this server, use command `[your_client].datasets`"
            ),
            value=result,
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
    ) -> DatasetPageView | DictTuple[str, Dataset]:
        """Get a Dataset"""
        datasets = self.stash.get_all_active(context.credentials).unwrap()

        for dataset in datasets:
            if context.server is not None:
                dataset.server_uid = context.server.id

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
    ) -> DatasetPageView | DictTuple[str, Dataset]:
        """Search a Dataset by name"""
        results = self.get_all(context)

        filtered_results = [
            dataset
            for dataset_name, dataset in results.items()
            if name in dataset_name and not dataset.to_be_deleted
        ]

        return _paginate_dataset_collection(
            filtered_results, page_size=page_size, page_index=page_index
        )

    @service_method(path="dataset.get_by_id", name="get_by_id")
    def get_by_id(self, context: AuthedServiceContext, uid: UID) -> Dataset:
        """Get a Dataset"""
        dataset = self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

        if context.server is not None:
            dataset.server_uid = context.server.id

        return dataset

    @service_method(path="dataset.get_by_action_id", name="get_by_action_id")
    def get_by_action_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> list[Dataset]:
        """Get Datasets by an Action ID"""
        datasets = self.stash.search_action_ids(context.credentials, uid=uid).unwrap()

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
    ) -> list[Asset]:
        """Get Assets by an Action ID"""
        datasets = self.get_by_action_id(context=context, uid=uid)

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
        unwrap_on_success=False,
    )
    def delete(
        self, context: AuthedServiceContext, uid: UID, delete_assets: bool = True
    ) -> SyftSuccess:
        """
        Soft delete: keep the dataset object, only remove the blob store entries
        After soft deleting a dataset, the user will not be able to
        see it using the `datasets.get_all` endpoint.
        Delete unique `dataset.name` key and leave UID, just rename it in case the
        user upload a new dataset with the same name.
        """
        # check if the dataset exists
        dataset = self.get_by_id(context=context, uid=uid)

        return_msg = []

        if delete_assets:
            # delete the dataset's assets
            for asset in dataset.asset_list:
                msg = (
                    f"ActionObject {asset.action_id} "
                    f"linked with Assset {asset.id} "
                    f"in Dataset {uid}"
                )

                context.server.services.action.delete(
                    context=context, uid=asset.action_id, soft_delete=True
                )

                logger.info(f"Successfully deleted {msg}")
                return_msg.append(f"Asset with id '{asset.id}' successfully deleted.")

        # soft delete the dataset object from the store
        dataset.name = f"_deleted_{dataset.name}_{uid}"
        dataset.to_be_deleted = True
        self.stash.update(context.credentials, dataset).unwrap()
        return_msg.append(f"Dataset with id '{uid}' successfully deleted.")
        return SyftSuccess(message="\n".join(return_msg))


TYPE_TO_SERVICE[Dataset] = DatasetService
SERVICE_TO_TYPES[DatasetService].update({Dataset})
