# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy.orm import Session

# relative
from . import Base


class StorageMetadata(Base):
    __tablename__ = "storage_metadata"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    length = Column(Integer())

    def __str__(self) -> str:
        return f"<StorageMetadata length: {self.length}>"


def get_metadata(db_session: Session) -> StorageMetadata:
    metadata = db_session.query(StorageMetadata).first()

    if metadata is None:
        metadata = StorageMetadata(length=0)
        db_session.add(metadata)
        db_session.commit()

    return metadata
