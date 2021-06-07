# grid relative
from .. import BaseModel
from .. import db


class JsonObject(BaseModel):
    __bind_key__ = "bin_store"
    __tablename__ = "json_object"

    id = db.Column(db.String(), primary_key=True)
    binary = db.Column(db.JSON())

    def __str__(self):
        return f"<JsonObject id: {self.id}>"
