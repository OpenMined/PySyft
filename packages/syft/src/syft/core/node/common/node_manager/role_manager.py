# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query

# relative
from ..exceptions import RoleNotFoundError
from ..node_table.roles import Role
from .database_manager import DatabaseManager


class RoleManager(DatabaseManager):
    schema = Role

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=RoleManager.schema, db=database)

    @property
    def ds_role(self) -> Role:
        return self.first(name="Data Scientist")

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
            can_make_data_requests=True,
            can_triage_data_requests=False,
            can_manage_privacy_budget=False,
            can_create_users=False,
            can_manage_users=False,
            can_edit_roles=False,
            can_manage_infrastructure=False,
            can_upload_data=False,
            can_upload_legal_document=False,
            can_edit_domain_settings=False,
        )

    def can_create_users(self, role_id: int) -> bool:
        role = super().first(id=role_id)
        if role:
            return role.can_create_users
        else:
            return False

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


class Singleton(type):
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args: List[Any], **kwargs: Dict[Any, Any]):  # type: ignore
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Make it a singleton class.
class NewRoleManager(metaclass=Singleton):
    role_dict: dict = {}

    def __init__(self) -> None:
        self._add_roles()

    def _add_roles(self) -> None:
        self.add(
            role_name="ds_role",
            name="Data Scientist",
            can_make_data_requests=True,
            can_triage_data_requests=False,
            can_manage_privacy_budget=False,
            can_create_users=False,
            can_manage_users=False,
            can_edit_roles=False,
            can_manage_infrastructure=False,
            can_upload_data=False,
            can_upload_legal_document=False,
            can_edit_domain_settings=False,
        )

        self.add(
            role_name="compliance_officer_role",
            name="Compliance Officer",
            can_make_data_requests=False,
            can_triage_data_requests=True,
            can_manage_privacy_budget=False,
            can_create_users=False,
            can_manage_users=True,
            can_edit_roles=False,
            can_manage_infrastructure=False,
            can_upload_data=False,
            can_upload_legal_document=False,
            can_edit_domain_settings=False,
        )

        self.add(
            role_name="admin_role",
            name="Administrator",
            can_make_data_requests=True,
            can_triage_data_requests=True,
            can_manage_privacy_budget=True,
            can_create_users=True,
            can_manage_users=True,
            can_edit_roles=True,
            can_manage_infrastructure=True,
            can_upload_data=True,
            can_upload_legal_document=True,
            can_edit_domain_settings=True,
        )

        self.add(
            role_name="owner_role",
            name="Owner",
            can_make_data_requests=True,
            can_triage_data_requests=True,
            can_manage_privacy_budget=True,
            can_create_users=True,
            can_manage_users=True,
            can_edit_roles=True,
            can_manage_infrastructure=True,
            can_upload_data=True,
            can_upload_legal_document=True,
            can_edit_domain_settings=True,
        )

    def add(self, role_name: str, **kwargs) -> None:  # type: ignore
        self.role_dict[role_name] = kwargs

    @property
    def ds_role(self) -> Role:
        role = self.role_dict.get("ds_role", None)
        if role is None:
            raise ValueError("Data Scientist role not populated.")
        return role

    @property
    def owner_role(self) -> Role:
        role = self.role_dict.get("owner_role", None)
        if role is None:
            raise ValueError("Owner role not populated.")
        return role

    @property
    def compliance_officer_role(self) -> Role:
        role = self.role_dict.get("compliance_officer_role", None)
        if role is None:
            raise ValueError("Compliance Officer role not populated.")
        return role

    @property
    def admin_role(self) -> Role:
        role = self.role_dict.get("admin_role", None)
        if role is None:
            raise ValueError("Admin role not populated.")
        return role

    def contain(self, name: str) -> bool:
        for _, v in self.role_dict.items():
            if v["name"] == name:
                return True
        return False
