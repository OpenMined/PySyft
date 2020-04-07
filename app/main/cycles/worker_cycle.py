import datetime
from .. import db


class WorkerCycle(db.Model):
    """ Relation between Workers and Cycles.
        Columns:
            id (Integer, Primary Key): Worker Cycle ID.
            cycle_id (Integer, ForeignKey): Cycle Foreign key that owns this worker cycle.
            worker_id (String, ForeignKey): Worker Foreign key that owns this worker cycle.
            request_key (String): unique token that permits downloading specific Plans, Protocols, etc.
    """

    __tablename__ = "__worker_cycle__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    request_key = db.Column(db.String())
    cycle_id = db.Column(db.BigInteger, db.ForeignKey("__cycle__.id"))
    worker_id = db.Column(db.String, db.ForeignKey("__worker__.id"))
    started_at = db.Column(db.DateTime(), default=datetime.datetime.utcnow())
    is_completed = db.Column(db.Boolean(), default=False)
    completed_at = db.Column(db.DateTime())
    diff = db.Column(db.LargeBinary)

    def __str__(self):
        f"<WorkerCycle id: {self.id}, fl_process: {self.fl_process_id}, cycle: {self.cycle_id}, worker: {self.worker}, request_key: {self.request_key}>"
