# grid relative
from .. import BaseModel
from .. import db


class StorageMetadata(BaseModel):
    __bind_key__ = "bin_store"
    __tablename__ = "storage_metadata"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    length = db.Column(db.Integer())

    def __str__(self):
        return f"<StorageMetadata length: {self.length}>"


def get_metadata(db):

    metadata = db.session.query(StorageMetadata).first()

    if metadata is None:
        metadata = StorageMetadata(length=0)
        db.session.add(metadata)
        db.session.commit()

    return metadata
