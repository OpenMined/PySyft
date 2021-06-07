# stdlib
from typing import List

# grid relative
from ..database.groups.groups import Group
from ..database.groups.usergroup import UserGroup
from ..exceptions import AuthorizationError
from ..exceptions import GroupNotFoundError
from ..exceptions import InvalidCredentialsError
from ..exceptions import MissingRequestKeyError
from ..exceptions import PyGridError
from ..exceptions import RoleNotFoundError
from ..exceptions import UserNotFoundError
from .database_manager import DatabaseManager


class GroupManager(DatabaseManager):

    schema = Group
    user_group_association_schema = UserGroup

    def __init__(self, database):
        self._schema = GroupManager.schema
        self._association_schema = GroupManager.user_group_association_schema
        self.db = database

    def first(self, **kwargs) -> Group:
        result = super().first(**kwargs)
        if not result:
            raise GroupNotFoundError
        return result

    def create(self, group_name: str, users: List = None):
        group_obj = self.register(name=group_name)
        if users:
            self._attach_users(group_id=group_obj.id, users=users)

    def update(self, group_id, group_name: str = None, users: List = None):
        if group_name:
            self.modify(query={"id": group_id}, values={"name": group_name})

        if users:
            self.delete_association(group=group_id)
            self._attach_users(group_id=group_id, users=users)

    def get_users(self, group_id: str):
        _associations = self.db.session.query(self._association_schema).filter_by(
            group=group_id
        )
        return [assoc.user for assoc in _associations]

    def get_groups(self, user_id: str):
        _associations = self.db.session.query(self._association_schema).filter_by(
            user=user_id
        )
        return [assoc.group for assoc in _associations]

    def contain_association(self, **kwargs):
        result = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).first()
        )
        if result:
            return True
        else:
            return False

    def update_user_association(self, user_id, groups):
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

    def _attach_users(self, group_id, users):
        for user in users:
            _user_group_association = self._association_schema(
                user=user, group=group_id
            )
            self.db.session.add(_user_group_association)

        self.db.session.commit()

    def delete_association(self, **kwargs):
        objects_to_delete = (
            self.db.session.query(self._association_schema).filter_by(**kwargs).all()
        )
        for obj in objects_to_delete:
            self.db.session.delete(obj)
        self.db.session.commit()
