# third party
from sqlalchemy import Column
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base


class JsonObject(Base):
    __tablename__ = "json_object"

    id = Column(String(), primary_key=True)
    binary = Column(JSON())

    def __str__(self) -> str:
        return f"<JsonObject id: {self.id}>"
