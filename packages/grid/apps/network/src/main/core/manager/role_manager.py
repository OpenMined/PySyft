# stdlib
from typing import List
from typing import Union

# grid relative
from ..database.roles.roles import Role
from ..exceptions import RoleNotFoundError
from .database_manager import DatabaseManager


class RoleManager(DatabaseManager):

    schema = Role

    def __init__(self, database):
        self._schema = RoleManager.schema
        self.db = database

    @property
    def user_role(self):
        return self.first(name="User")

    @property
    def owner_role(self):
        return self.first(name="Owner")

    @property
    def compliance_officer_role(self):
        return self.first(name="Compliance Officer")

    @property
    def admin_role(self):
        return self.first(name="Administrator")

    @property
    def common_roles(self):
        return self.db.session.query(self._schema).filter_by(
            can_triage_requests=False,
            can_edit_settings=False,
            can_create_users=False,
            can_create_groups=False,
            can_upload_data=False,
            can_edit_roles=False,
            can_manage_infrastructure=False,
        )

    @property
    def org_roles(self):
        return self.db.session.query(self._schema).except_(self.common_roles)

    def first(self, **kwargs) -> Union[None, List]:
        result = super().first(**kwargs)
        if not result:
            raise RoleNotFoundError
        return result

    def query(self, **kwargs) -> Union[None, List]:
        results = super().query(**kwargs)
        if len(results) == 0:
            raise RoleNotFoundError
        return results

    def set(self, role_id, params):
        if self.contain(id=role_id):
            self.modify({"id": role_id}, params)
        else:
            raise RoleNotFoundError
