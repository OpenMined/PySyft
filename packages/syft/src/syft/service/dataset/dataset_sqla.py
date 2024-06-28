# stdlib
from datetime import datetime
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import Column, UniqueConstraint
from sqlalchemy import ForeignKey
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from ..action.action_permissions import ActionPermission


engine = create_engine("sqlite://")


class Base(DeclarativeBase):
    pass


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(server_default=sa.func.now())
    modified_at: Mapped[datetime] = mapped_column(
        server_default=sa.func.now(), server_onupdate=sa.func.now()
    )
    modified_by: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)


contributor_asset_association_table = Table(
    "contributor_asset_association_table",
    Base.metadata,
    Column("contributor_id", ForeignKey("contributor.id")),
    Column("asset_action_id", ForeignKey("asset.action_id")),
)

dataset_asset_association_table = Table(
    "dataset_asset_association_table",
    Base.metadata,
    Column("dataset_id", ForeignKey("dataset.id")),
    Column("asset_action_id", ForeignKey("asset.action_id")),
)

dataset_contributor_association_table = Table(
    "dataset_contributor_association_table",
    Base.metadata,
    Column("dataset_id", ForeignKey("dataset.id")),
    Column("contributor_id", ForeignKey("contributor.id")),
)


class Contributor(CommonMixin, Base):
    name: Mapped[str]
    role: Mapped[str | None]
    email: Mapped[str]
    phone: Mapped[str | None]
    note: Mapped[str | None]

    assets: Mapped[list["Asset"]] = relationship(
        back_populates="contributors", secondary=contributor_asset_association_table
    )
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="contributors", secondary=dataset_contributor_association_table
    )
    uploaded_datasets: Mapped[list["Dataset"]] = relationship("Dataset")


class Asset(CommonMixin, Base):
    action_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    node_uid: Mapped[uuid.UUID]
    name: Mapped[str]
    description: Mapped[str | None]
    contributors: Mapped[list["Contributor"]] = relationship(
        back_populates="assets", secondary=contributor_asset_association_table
    )
    mock_is_real: Mapped[bool] = mapped_column(default=False)
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="asset_list", secondary=dataset_asset_association_table
    )


class Dataset(CommonMixin, Base):
    name: Mapped[str]
    description: Mapped[str | None]
    node_uid: Mapped[uuid.UUID | None]
    asset_list: Mapped[list[Asset]] = relationship(
        back_populates="datasets", secondary=dataset_asset_association_table
    )
    contributors: Mapped[list["Contributor"]] = relationship(
        back_populates="datasets", secondary=dataset_contributor_association_table
    )
    uploader_id: Mapped[int] = mapped_column(ForeignKey("contributor.id"))
    uploader: Mapped["Contributor"] = relationship(back_populates="uploaded_datasets")

    permissions: Mapped[list["DatasetPermission"]] = relationship("DatasetPermission")
    storage_permissions: Mapped[list["DatasetStoragePermission"]] = relationship(
        "DatasetStoragePermission"
    )


class DatasetPermission(Base):
    __tablename__ = "dataset_permission"

    id: Mapped[int] = mapped_column(primary_key=True)
    object_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset.id"))
    verify_key: Mapped[str | None]
    permission: Mapped[ActionPermission]


class DatasetStoragePermission(Base):
    __tablename__ = "dataset_storage_permission"

    id: Mapped[int] = mapped_column(primary_key=True)
    object_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset.id"))
    node_uid: Mapped[uuid.UUID]
