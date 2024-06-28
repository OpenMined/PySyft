# stdlib
from collections.abc import Collection
from collections.abc import Sequence

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ...util.telemetry import instrument
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
from .dataset_sqla_stash import DatasetSQLStash


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
@serializable()
class DatasetService(AbstractService):
    store: DocumentStore
    stash: DatasetSQLStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DatasetSQLStash()

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
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(
            message=f"Dataset uploaded to '{context.node.name}'. "
            f"To see the datasets uploaded by a client on this node, use command `[your_client].datasets`"
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
            if context.node is not None:
                dataset.node_uid = context.node.id

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
            dataset for dataset_name, dataset in results.items() if name in dataset_name
        ]

        return _paginate_dataset_collection(
            filtered_results, page_size=page_size, page_index=page_index
        )

    @service_method(path="dataset.get_by_id", name="get_by_id")
    def get_by_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Get a Dataset"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            dataset = result.ok()
            if context.node is not None:
                dataset.node_uid = context.node.id
            return dataset
        return SyftError(message=result.err())

    @service_method(path="dataset.get_by_action_id", name="get_by_action_id")
    def get_by_action_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> list[Dataset] | SyftError:
        """Get Datasets by an Action ID"""
        result = self.stash.search_action_ids(context.credentials, uid=uid)
        if result.is_ok():
            datasets = result.ok()
            for dataset in datasets:
                if context.node is not None:
                    dataset.node_uid = context.node.id
            return datasets
        return SyftError(message=result.err())

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
        return [
            asset
            for dataset in datasets
            for asset in dataset.asset_list
            if asset.action_id == uid
        ]

    @service_method(
        path="dataset.delete_by_uid",
        name="delete_by_uid",
        roles=DATA_OWNER_ROLE_LEVEL,
        warning=HighSideCRUDWarning(confirmation=True),
    )
    def delete_dataset(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())


TYPE_TO_SERVICE[Dataset] = DatasetService
SERVICE_TO_TYPES[DatasetService].update({Dataset})
