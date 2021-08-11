# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class Role(Base):
    __tablename__ = "role"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    name = Column(String(255))
    can_triage_requests = Column(Boolean(), default=False)
    can_edit_settings = Column(Boolean(), default=False)
    can_create_users = Column(Boolean(), default=False)
    can_create_groups = Column(Boolean(), default=False)
    can_edit_roles = Column(Boolean(), default=False)
    can_manage_infrastructure = Column(Boolean(), default=False)
    can_upload_data = Column(Boolean(), default=False)

    def __str__(self) -> str:
        return (
            f"<Role id: {self.id}, name: {self.name}, "
            f"can_triage_requests: {self.can_triage_requests}, "
            f"can_edit_settings: {self.can_edit_settings}, "
            f"can_create_users: {self.can_create_users}, "
            f"can_create_groups: {self.can_create_groups}, "
            f"can_edit_roles: {self.can_edit_roles}, "
            f"can_manage_infrastructure: {self.can_manage_infrastructure}>, "
            f"can_upload_data: {self.can_upload_data}>"
        )


def create_role(
    name: str,
    can_triage_requests: bool,
    can_edit_settings: bool,
    can_create_users: bool,
    can_create_groups: bool,
    can_edit_roles: bool,
    can_manage_infrastructure: bool,
    can_upload_data: bool,
) -> Role:
    new_role = Role(
        name=name,  # type:ignore
        can_triage_requests=can_triage_requests,
        can_edit_settings=can_edit_settings,
        can_create_users=can_create_users,
        can_create_groups=can_create_groups,
        can_edit_roles=can_edit_roles,
        can_manage_infrastructure=can_manage_infrastructure,
        can_upload_data=can_upload_data,
    )
    return new_role
