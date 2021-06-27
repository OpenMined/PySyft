# syft relative
from .. import BaseModel
from .. import db


class ObjectMetadata(BaseModel):
    __tablename__ = "obj_metadata"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # TODO: @Ionesio investigate the difference
    obj = db.Column(db.String(3072), db.ForeignKey("bin_object.id", ondelete="CASCADE"))
    # obj = db.Column(db.String(3072), db.ForeignKey("bin_object.id", ondelete='SET NULL'), nullable=True)

    tags = db.Column(db.JSON())
    description = db.Column(db.String())
    read_permissions = db.Column(db.JSON())
    search_permissions = db.Column(db.JSON())
