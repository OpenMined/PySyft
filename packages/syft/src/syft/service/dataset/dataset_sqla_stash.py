# third party
import uuid
from result import Err, Ok
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from result import Result
from sqlalchemy import select
from sqlalchemy.orm import Session
from syft.service.action.action_permissions import (
    COMPOUND_ACTION_PERMISSION,
    ActionObjectPermission,
    ActionPermission,
    StoragePermission,
)

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

    def set(
        self,
        credentials: SyftVerifyKey,
        dataset: Dataset,
    ) -> Result[Dataset, str]:
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
            try:
                session.add(dataset_sqla_obj)
                session.flush()

                add_permission_res = self.add_permission(
                    session=session,
                    permission=ActionObjectPermission(
                        uid=UID(dataset_sqla_obj.id),
                        permission=ActionPermission.READ,
                        credentials=credentials,
                    ),
                )
                if add_permission_res.is_err():
                    session.rollback()
                    return add_permission_res

                add_storage_permission_res = self.add_storage_permission(
                    session=session,
                    permission=StoragePermission(
                        uid=UID(dataset_sqla_obj.id),
                        node_uid=UID(dataset.node_uid),
                    ),
                )
                if add_storage_permission_res.is_err():
                    session.rollback()
                    return add_storage_permission_res

                session.commit()
            except IntegrityError as e:
                session.rollback()
                return Err(str(e))

        return Ok(dataset)

    def add_storage_permission(
        self,
        session: Session,
        permission: StoragePermission,
    ) -> Result[None, str]:
        new_permission = dataset_sqla.DatasetStoragePermission(
            object_uid=permission.uid.value,
            node_uid=permission.node_uid.value,
        )
        session.add(new_permission)
        return Ok(None)

    def add_permission(
        self,
        session: Session,
        permission: ActionObjectPermission,
    ) -> Result[None, str]:
        new_permission = dataset_sqla.DatasetPermission(
            object_uid=permission.uid.value,
            verify_key=permission.credentials.verify,
            permission=permission.permission,
        )
        session.add(new_permission)
        return Ok(None)

    def has_permission(
        self,
        session: Session,
        permission: ActionObjectPermission,
    ) -> bool:
        object_uid = permission.uid.value
        permission_type = permission.permission
        credentials = permission.credentials

        if permission_type in COMPOUND_ACTION_PERMISSION:
            stmt = select(dataset_sqla.DatasetPermission).where(
                dataset_sqla.DatasetPermission.object_uid == object_uid,
                dataset_sqla.DatasetPermission.permission == permission_type,
            )

        else:
            if credentials is None:
                raise ValueError(
                    "Credentials must be provided for non-compound permissions."
                )
            stmt = select(dataset_sqla.DatasetPermission).where(
                dataset_sqla.DatasetPermission.object_uid == object_uid,
                dataset_sqla.DatasetPermission.verify_key == credentials.verify,
                dataset_sqla.DatasetPermission.permission == permission_type,
            )

        result = session.execute(stmt).scalar_one_or_none()
        return result is not None

    def _get_all_with_permission(
        self, session: Session, credentials: SyftVerifyKey, permission: ActionPermission
    ) -> Result[list[Dataset], str]:
        stmt = (
            select(dataset_sqla.Dataset)
            .join(
                dataset_sqla.DatasetPermission,
                dataset_sqla.Dataset.id == dataset_sqla.DatasetPermission.object_uid,
            )
            .where(
                dataset_sqla.DatasetPermission.verify_key == credentials.verify,
                dataset_sqla.DatasetPermission.permission == permission,
            )
        )

        return session.scalars(stmt).all()

    def get_all(self, credentials: SyftVerifyKey) -> Result[list[Dataset], str]:
        with Session(dataset_sqla.engine) as session:
            results = self._get_all_with_permission(
                session, credentials, ActionPermission.READ
            )

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
