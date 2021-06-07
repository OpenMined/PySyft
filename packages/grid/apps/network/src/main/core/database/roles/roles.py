# grid relative
from .. import BaseModel
from .. import db


class Role(BaseModel):
    __tablename__ = "role"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String(255))
    can_edit_settings = db.Column(db.Boolean(), default=False)
    can_create_users = db.Column(db.Boolean(), default=False)
    can_edit_roles = db.Column(db.Boolean(), default=False)
    can_manage_infrastructure = db.Column(db.Boolean(), default=False)

    def __str__(self):
        return (
            f"<Role id: {self.id}, name: {self.name}, "
            f"can_edit_settings: {self.can_edit_settings}, "
            f"can_create_users: {self.can_create_users}, "
            f"can_edit_roles: {self.can_edit_roles}, "
            f"can_manage_infrastructure: {self.can_manage_infrastructure}>, "
        )


def create_role(
    name,
    can_edit_settings,
    can_create_users,
    can_edit_roles,
    can_manage_infrastructure,
):
    new_role = Role(
        name=name,
        can_edit_settings=can_edit_settings,
        can_create_users=can_create_users,
        can_edit_roles=can_edit_roles,
        can_manage_infrastructure=can_manage_infrastructure,
    )
    return new_role
