from .. import db


class Protocol(db.Model):
    """ Protocol table that represents Syft Protocols.
        Columns:
            id (Integer, Primary Key): Protocol ID.
            name (String): protocol name.
            value: String  (List of operations)
            value_ts: String (TorchScript)
            fl_process_id (Integer, Foreign Key) : Reference to FL Process.
    """

    __tablename__ = "__protocol__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String())
    value = db.Column(db.LargeBinary)
    value_ts = db.Column(db.LargeBinary)
    fl_process_id = db.Column(db.BigInteger, db.ForeignKey("__fl_process__.id"))

    def __str__(self):
        return f"<Protocol id: {self.id}, values: {self.value}, torchscript: {self.value_ts}>"
