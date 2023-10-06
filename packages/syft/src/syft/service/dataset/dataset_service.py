# stdlib
from itertools import islice
from typing import List
from typing import Optional
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.tupledict import TupleDict
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
from .dataset import Asset
from .dataset import CreateDataset
from .dataset import Dataset
from .dataset import DatasetPageView
from .dataset_stash import DatasetStash


def _create_dataset_page_view(
    results: TupleDict[str, Dataset],
    page_size: Optional[int] = 0,
    page_index: Optional[int] = 0,
) -> DatasetPageView:
    if page_size <= 0 or page_size is None:
        return results

    # If chunk size is defined, then split list into evenly sized chunks
    total = len(results)
    page_index = 0 if page_index is None else page_index

    if page_size > total or page_index >= total // page_size or page_index < 0:
        pass
    else:
        results = TupleDict(
            islice(
                results.items(),
                page_size * page_index,
                min(page_size * (page_index + 1), total),
            )
        )

    return DatasetPageView(datasets=results, total=total)


@instrument
@serializable()
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
    ) -> Union[SyftSuccess, SyftError]:
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
        page_size: Optional[int] = 0,
        page_index: Optional[int] = 0,
    ) -> Union[DatasetPageView, TupleDict[str, Dataset], SyftError]:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        if not result.is_ok():
            return SyftError(message=result.err())

        datasets = result.ok()

        results = TupleDict()
        for dataset in datasets:
            dataset.node_uid = context.node.id
            results[dataset.name] = dataset

        return (
            results
            if page_size <= 0 or page_size is None
            else _create_dataset_page_view(
                results, page_size=page_size, page_index=page_index
            )
        )

    @service_method(
        path="dataset.search", name="search", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def search(
        self,
        context: AuthedServiceContext,
        name: str,
        page_size: Optional[int] = 0,
        page_index: Optional[int] = 0,
    ) -> Union[DatasetPageView, SyftError]:
        """Search a Dataset by name"""
        results = self.get_all(context)

        if isinstance(results, SyftError):
            return results

        filtered_results = TupleDict(
            (dataset_name, dataset)
            for dataset_name, dataset in results.items()
            if name in dataset_name
        )

        return _create_dataset_page_view(
            filtered_results, page_size=page_size, page_index=page_index
        )

    @service_method(path="dataset.get_by_id", name="get_by_id")
    def get_by_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get a Dataset"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            dataset = result.ok()
            dataset.node_uid = context.node.id
            return dataset
        return SyftError(message=result.err())

    @service_method(path="dataset.get_by_action_id", name="get_by_action_id")
    def get_by_action_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[List[Dataset], SyftError]:
        """Get Datasets by an Action ID"""
        result = self.stash.search_action_ids(context.credentials, uid=uid)
        if result.is_ok():
            datasets = result.ok()
            for dataset in datasets:
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
    ) -> Union[List[Asset], SyftError]:
        """Get Assets by an Action ID"""
        datasets = self.get_by_action_id(context=context, uid=uid)
        assets = []
        if isinstance(datasets, list):
            for dataset in datasets:
                for asset in dataset.asset_list:
                    if asset.action_id == uid:
                        assets.append(asset)
            return assets
        elif isinstance(datasets, SyftError):
            return datasets
        return []

    @service_method(
        path="dataset.delete_by_id",
        name="dataset_delete_by_id",
        warning=HighSideCRUDWarning(confirmation=True),
    )
    def delete_dataset(self, context: AuthedServiceContext, uid: UID):
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())


TYPE_TO_SERVICE[Dataset] = DatasetService
SERVICE_TO_TYPES[DatasetService].update({Dataset})
