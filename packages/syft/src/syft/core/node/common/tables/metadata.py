# grid relative
# syft relative
from . import Base
from sqlalchemy import Column, Integer, String, ForeignKey


class StorageMetadata(Base):
    __tablename__ = "storage_metadata"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    length = Column(Integer())

    def __str__(self):
        return f"<StorageMetadata length: {self.length}>"


def get_metadata(db_session):

    metadata = db_session.query(StorageMetadata).first()

    if metadata is None:
        metadata = StorageMetadata(length=0)
        db_session.add(metadata)
        db_session.commit()

    return metadata
