from .. import BaseModel, db


class SetupConfig(BaseModel):
    __tablename__ = "setup"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    cloud_credentials = db.Column(db.String(255))

    def __str__(self):
        return f"<Setup id: {self.id}, cloud_credentials: {self.cloud_credentials}>"
