# stdlib

# stdlib

# third party
from result import Ok
from result import Result
from sqlalchemy import select
from sqlalchemy.orm import Session

# relative
from . import dataset
from . import dataset_sqla
from ...node.credentials import SyftVerifyKey
from ...types.uid import UID
from ...util.misc_objs import MarkdownDescription
from .dataset import Dataset
from .dataset import DatasetUpdate


class DatasetSQLStash:
    def __init__(self) -> None:
        pass

    def set(self, credentials: SyftVerifyKey, dataset: Dataset) -> Result[Dataset, str]:
        assets = [
            dataset_sqla.Asset(
                name=asset.name,
                action_id=asset.action_id.value,
                node_uid=asset.node_uid.value,
            )
            for asset in dataset.asset_list
        ]
        contributors = [
            dataset_sqla.Contributor(
                name=contributor.name,
                email=contributor.email,
                phone=contributor.phone,
                note=contributor.note,
            )
            for contributor in dataset.contributors
        ]
        contributor = dataset_sqla.Contributor(
            name=credentials.verify,
            email="",
            phone="",
            note="",
        )
        dataset_sqla_obj = dataset_sqla.Dataset(
            name=dataset.name,
            description=dataset.description.text,
            # tags=dataset.tags,
            # action_ids=dataset.action_ids(),
            asset_list=assets,
            contributors=contributors,
            uploader=contributor,
        )
        # dataset_sqla_obj.asset_list.extend(assets)

        with Session(dataset_sqla.engine) as session:
            session.add(dataset_sqla_obj)
            session.commit()
        return Ok(dataset)

    def get_all(self, credentials: SyftVerifyKey) -> Result[list[Dataset], str]:
        with Session(dataset_sqla.engine) as session:
            stmt = select(dataset_sqla.Dataset)
            results = session.scalars(stmt).all()

            syft_datasets = []
            for dtset in results:
                syft_dataset = Dataset(
                    name=dtset.name,
                    description=MarkdownDescription(text=dtset.description),
                    asset_list=[
                        dataset.Asset(
                            name=asset.name,
                            action_id=UID(asset.action_id),
                            node_uid=UID(asset.node_uid),
                        )
                        for asset in dtset.asset_list
                    ],
                    contributors=[
                        dataset.Contributor(
                            name=contributor.name,
                            email=contributor.email,
                            phone=contributor.phone,
                            note=contributor.note,
                        )
                        for contributor in dtset.contributors
                    ],
                    uploader=dataset.Contributor(
                        name=dtset.uploader.name,
                        email=dtset.uploader.email,
                        phone=dtset.uploader.phone,
                        note=dtset.uploader.note,
                    ),
                )
                syft_datasets.append(syft_dataset)
            return Ok(syft_datasets)

    def update(
        self,
        credentials: SyftVerifyKey,
        dataset_update: DatasetUpdate,
        has_permission: bool = False,
    ) -> Result[Dataset, str]:
        res = self.check_type(dataset_update, DatasetUpdate)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().update(credentials=credentials, obj=res.ok())

    def search_action_ids(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[list[Dataset], str]:
        qks = QueryKeys(qks=[ActionIDsPartitionKey.with_obj(uid)])
        return self.query_all(credentials=credentials, qks=qks)
