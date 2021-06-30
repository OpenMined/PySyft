# grid relative
# syft relative
from . import Base
from sqlalchemy import Column, String, JSON


class JsonObject(Base):
    __tablename__ = "json_object"

    id = db.Column(String(), primary_key=True)
    binary = db.Column(JSON())

    def __str__(self):
        return f"<JsonObject id: {self.id}>"
