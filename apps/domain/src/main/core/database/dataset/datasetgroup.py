from .. import BaseModel, db


class DatasetGroup(BaseModel):
    __tablename__ = "datasetgroup"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    bin_object = db.Column(db.String(), db.ForeignKey("bin_object.id"))
    dataset = db.Column(db.String(), db.ForeignKey("json_object.id"))

    def __str__(self):
        return (
            f"<DatasetGroup id: {self.id}, bin_object: {self.bin_object}, "
            f"dataset: {self.dataset}>"
        )
