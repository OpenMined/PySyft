from ... import BaseModel, db


class Role(BaseModel):
    __tablename__ = "role"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String())
    can_triage_jobs = db.Column(db.Boolean())
    can_edit_settings = db.Column(db.Boolean())
    can_create_users = db.Column(db.Boolean())
    can_create_groups = db.Column(db.Boolean())
    can_edit_roles = db.Column(db.Boolean())
    can_manage_infrastructure = db.Column(db.Boolean())

    def __str__(self):
        return (
            f"<Role id: {self.id}, name: {self.name}, "
            f"can_triage_jobs: {self.can_triage_jobs}, "
            f"can_edit_settings: {self.can_edit_settings}, "
            f"can_create_users: {self.can_create_users}, "
            f"can_create_groups: {self.can_create_groups}, "
            f"can_edit_roles: {self.can_edit_roles}, "
            f"can_manage_infrastructure: {self.can_manage_infrastructure}>"
        )
