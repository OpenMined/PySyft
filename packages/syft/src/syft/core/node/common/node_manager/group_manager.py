# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ..exceptions import GroupNotFoundError
from ..node_table.groups import Group
from ..node_table.usergroup import UserGroup
from .database_manager import DatabaseManager


class GroupManager(DatabaseManager):

    schema = Group
    user_group_association_schema = UserGroup

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=GroupManager.schema, db=database)
        self._association_schema = GroupManager.user_group_association_schema

    def first(self, **kwargs: Any) -> Group:
        result = super().first(**kwargs)
        if not result:
            raise GroupNotFoundError
        return result

    def create(self, group_name: str, users: Optional[List] = None) -> None:
        group_obj = self.register(name=group_name)
        if users:
            self._attach_users(group_id=group_obj.id, users=users)

    def update(
        self,
        group_id: int,
        group_name: Optional[str] = None,
        users: Optional[List] = None,
    ) -> None:
        if group_name:
            self.modify(query={"id": group_id}, values={"name": group_name})

        if users:
            self.delete_association(group=group_id)
            self._attach_users(group_id=group_id, users=users)

    def get_users(self, group_id: int) -> List[int]:
        _associations = self.db.session.query(self._association_schema).filter_by(
            group=group_id
        )
        return [assoc.user for assoc in _associations]

    def get_groups(self, user_id: int) -> List[int]:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        _associations = session_local.query(self._association_schema).filter_by(
            user=user_id
        )
        return [assoc.group for assoc in _associations]

    def contain_association(self, **kwargs: Any) -> bool:
        result = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).first()
        )
        return result is not None

    def update_user_association(self, user_id: int, groups: List[int]) -> None:
        # Delete all previous group associations with this user_id
        self.delete_association(user=user_id)
        # Create new ones
        for group_id in groups:
            # Check if group exists
            if self.contain(id=group_id):
                _user_group_association = self._association_schema(
                    user=user_id, group=group_id
                )
                self.db.session.add(_user_group_association)

        self.db.session.commit()

    def _attach_users(self, group_id: int, users: List[int]) -> None:
        for user in users:
            _user_group_association = self._association_schema(
                user=user, group=group_id
            )
            self.db.session.add(_user_group_association)

        self.db.session.commit()

    def delete_association(self, **kwargs: Any) -> None:
        objects_to_delete = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).all()
        )
        for obj in objects_to_delete:
            self.db.session.delete(obj)
        self.db.session.commit()
