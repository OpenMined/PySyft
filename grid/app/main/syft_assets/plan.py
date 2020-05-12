from .. import db


class Plan(db.Model):
    """ Plan table that represents Syft Plans.
        Columns:
            id (Integer, Primary Key): Plan ID.
            name (String): Plan name.
            value (Binary): String  (List of operations)
            value_ts (Binary): String (TorchScript)
            is_avg_plan (Boolean) : Boolean flag to indicate if it is the avg plan
            fl_process_id (Integer, Foreign Key) : Reference to FL Process.
    """

    __tablename__ = "__plan__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String())
    value = db.Column(db.LargeBinary)
    value_ts = db.Column(db.LargeBinary)
    is_avg_plan = db.Column(db.Boolean, default=False)
    fl_process_id = db.Column(db.BigInteger, db.ForeignKey("__fl_process__.id"))

    def __str__(self):
        return (
            f"<Plan id: {self.id}, value: {self.value}, torchscript: {self.value_ts}>"
        )
