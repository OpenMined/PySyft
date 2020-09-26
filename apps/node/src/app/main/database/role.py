from ... import BaseModel, db


class Role(BaseModel):
    __tablename__ = "role"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String(255))
    can_triage_requests = db.Column(db.Boolean())
    can_edit_settings = db.Column(db.Boolean())
    can_create_users = db.Column(db.Boolean())
    can_create_groups = db.Column(db.Boolean())
    can_edit_roles = db.Column(db.Boolean())
    can_manage_infrastructure = db.Column(db.Boolean())
    can_upload_data = db.Column(db.Boolean())

    def __str__(self):
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
    name,
    can_triage_requests,
    can_edit_settings,
    can_create_users,
    can_create_groups,
    can_edit_roles,
    can_manage_infrastructure,
    can_upload_data,
):
    new_role = Role(
        name=name,
        can_triage_requests=can_triage_requests,
        can_edit_settings=can_edit_settings,
        can_create_users=can_create_users,
        can_create_groups=can_create_groups,
        can_edit_roles=can_edit_roles,
        can_manage_infrastructure=can_manage_infrastructure,
        can_upload_data=can_upload_data,
    )
    return new_role
