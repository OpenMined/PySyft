# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer

# relative
from . import Base


class UserGroup(Base):
    __tablename__ = "usergroup"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    user = Column(Integer, ForeignKey("syft_user.id"))
    group = Column(Integer, ForeignKey("group.id"))

    def __str__(self) -> str:
        return f"<UserGroup id: {self.id}, user: {self.user}, " f"group: {self.group}>"
