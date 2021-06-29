# grid relative
# syft relative
from . import Base
from sqlalchemy import Boolean, Column, Integer, String, ForeignKey


class UserGroup(Base):
    __tablename__ = "usergroup"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    user = Column(Integer, ForeignKey("syft_user.id"))
    group = Column(Integer, ForeignKey("group.id"))

    def __str__(self):
        return f"<UserGroup id: {self.id}, user: {self.user}, " f"group: {self.group}>"
