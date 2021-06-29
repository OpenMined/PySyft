# grid relative
# syft relative
from . import Base
from sqlalchemy import Boolean, Column, Integer, String

class Group(Base):
    __tablename__ = "group"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    name = Column(String(255))

    def __str__(self):
        return f"<Group id: {self.id}, name: {self.name}>"
