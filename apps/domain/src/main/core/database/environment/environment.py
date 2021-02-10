from .. import BaseModel, db


class Environment(BaseModel):
    __tablename__ = "environment"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    address = db.Column(db.String(255))
    memory = db.Column(db.String(255))
    instance = db.Column(db.String(255))
    gpu = db.Column(db.String(255))

    def __str__(self):
        return f"<Group id: {self.id}, name: {self.address}, memory: {self.memory}, instance: {self.instance}, gpu: {self.gpu}>"
