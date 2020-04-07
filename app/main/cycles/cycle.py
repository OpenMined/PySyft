from .. import db


class Cycle(db.Model):
    """ Cycle table.
        Columns:
            id (Integer, Primary Key): Cycle ID.
            start (TIME): Start time.
            sequence (Integer): Cycle's sequence number.
            version (String) : Cycle Version.
            end (TIME): End time.
            worker_cycles (WorkerCycle): Relationship between workers and cycles (One to many).
            fl_process_id (Integer,ForeignKey): Federated learning ID that owns this cycle.
    """

    __tablename__ = "__cycle__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    start = db.Column(db.DateTime())
    end = db.Column(db.DateTime())
    sequence = db.Column(db.BigInteger())
    version = db.Column(db.String())
    worker_cycles = db.relationship("WorkerCycle", backref="cycle")
    fl_process_id = db.Column(db.BigInteger, db.ForeignKey("__fl_process__.id"))
    is_completed = db.Column(db.Boolean, default=False)

    def __str__(self):
        return f"< Cycle id : {self.id}, sequence: {self.sequence}, start: {self.start}, end: {self.end}, fl_process_id: {self.fl_process_id}, is_completed: {self.is_completed}>"
