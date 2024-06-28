from datetime import datetime
from typing import List
from typing import Optional
import uuid

from click import secho
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
import sqlalchemy as sa
from sqlalchemy import Uuid

from sqlalchemy import create_engine
from sqlalchemy import Table
from sqlalchemy import ForeignKey
from sqlalchemy import Column

engine = create_engine("sqlite://", echo=True)


class Base(DeclarativeBase):
    pass


contributor_asset_association_table = Table(
    "contributor_asset_association_table",
    Base.metadata,
    Column("contributor_id", ForeignKey("contributors.id")),
    Column("asset_action_id", ForeignKey("assets.action_id")),
)

dataset_asset_association_table = Table(
    "dataset_asset_association_table",
    Base.metadata,
    Column("dataset_id", ForeignKey("datasets.id")),
    Column("asset_action_id", ForeignKey("assets.action_id")),
)

dataset_contributor_association_table = Table(
    "dataset_contributor_association_table",
    Base.metadata,
    Column("dataset_id", ForeignKey("datasets.id")),
    Column("contributor_id", ForeignKey("contributors.id")),
)


class Contributor(Base):
    __tablename__ = "contributors"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    role: Mapped[Optional[str]]
    email: Mapped[str]
    phone: Mapped[Optional[str]]
    note: Mapped[Optional[str]]
    assets: Mapped[List["Asset"]] = relationship(
        back_populates="contributors", secondary=contributor_asset_association_table
    )
    datasets: Mapped[List["Dataset"]] = relationship(
        back_populates="contributors", secondary=dataset_contributor_association_table
    )
    uploaded_datasets: Mapped[List["Dataset"]] = relationship("Dataset")


class Asset(Base):
    __tablename__ = "assets"
    action_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    node_uid: Mapped[uuid.UUID]
    name: Mapped[str]
    description: Mapped[Optional[str]]
    contributors: Mapped[List["Contributor"]] = relationship(
        back_populates="assets", secondary=contributor_asset_association_table
    )
    mock_is_real: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=sa.func.now())
    datasets: Mapped[List["Dataset"]] = relationship(
        back_populates="asset_list", secondary=dataset_asset_association_table
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    node_uid: Mapped[Optional[uuid.UUID]]
    asset_list: Mapped[List[Asset]] = relationship(
        back_populates="datasets", secondary=dataset_asset_association_table
    )
    contributors: Mapped[List["Contributor"]] = relationship(
        back_populates="datasets", secondary=dataset_contributor_association_table
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sa.func.now(), onupdate=sa.func.now()
    )
    created_at: Mapped[datetime] = mapped_column(server_default=sa.func.now())
    uploader_id: Mapped[int] = mapped_column(ForeignKey("contributors.id"))
    uploader: Mapped["Contributor"] = relationship(back_populates="uploaded_datasets")
