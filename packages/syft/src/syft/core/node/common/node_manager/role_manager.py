# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query

# syft absolute
from syft.core.node.common.node_table.roles import Role

# relative
from ..exceptions import RoleNotFoundError
from .database_manager import DatabaseManager


class RoleManager(DatabaseManager):
    schema = Role

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=RoleManager.schema, db=database)

    @property
    def user_role(self) -> Role:
        return self.first(name="User")

    @property
    def owner_role(self) -> Role:
        return self.first(name="Owner")

    @property
    def compliance_officer_role(self) -> Role:
        return self.first(name="Compliance Officer")

    @property
    def admin_role(self) -> Role:
        return self.first(name="Administrator")

    def _common_roles(self) -> Query:
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
    def common_roles(self) -> List[Role]:
        return self._common_roles().all()

    @property
    def org_roles(self) -> List[Role]:
        return self.db.session.query(self._schema).except_(self._common_roles).all()

    def first(self, **kwargs: Any) -> Role:
        result = super().first(**kwargs)
        if not result:
            raise RoleNotFoundError
        return result

    def query(self, **kwargs: Any) -> Query:
        results = super().query(**kwargs)
        if len(results) == 0:
            raise RoleNotFoundError
        return results

    def set(self, role_id: int, params: Dict[Any, Any]) -> None:
        if self.contain(id=role_id):
            self.modify({"id": role_id}, params)
        else:
            raise RoleNotFoundError
