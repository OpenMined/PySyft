from ... import BaseModel, db


class Plan(BaseModel):
    """Plan table that represents Syft Plans.

    Columns:
        id (Integer, Primary Key): Plan ID.
        name (String): Plan name.
        value (Binary): String  (List of operations)
        value_ts (Binary): String (TorchScript)
        is_avg_plan (Boolean) : Boolean flag to indicate if it is the avg plan
        fl_process_id (Integer, Foreign Key) : Reference to FL Process.
    """

    __tablename__ = "model_centric_plan"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255))
    value = db.Column(db.LargeBinary)
    value_ts = db.Column(db.LargeBinary)
    value_tfjs = db.Column(db.LargeBinary)
    is_avg_plan = db.Column(db.Boolean, default=False)
    fl_process_id = db.Column(db.Integer, db.ForeignKey("model_centric_fl_process.id"))

    def __str__(self):
        return (
            f"<Plan id: {self.id}, value: {self.value}, torchscript: {self.value_ts}>"
        )
