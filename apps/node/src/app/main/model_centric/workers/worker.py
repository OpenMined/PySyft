from ... import BaseModel, db


class Worker(BaseModel):
    """Web / Mobile worker table.

    Columns:
        id (String, Primary Key): Worker's ID.
        ping (Float): Ping rate.
        avg_download (Float): Download rate.
        avg_upload (Float): Upload rate.
        worker_cycles (WorkerCycle): Relationship between workers and cycles (One to many).
    """

    __tablename__ = "model_centric_worker"

    id = db.Column(db.String, primary_key=True)
    ping = db.Column(db.Float)
    avg_download = db.Column(db.Float)
    avg_upload = db.Column(db.Float)
    worker_cycle = db.relationship("WorkerCycle", backref="worker")

    def __str__(self):
        return f"<Worker id: {self.id}, ping : {self.ping}, download: {self.avg_download}, upload: {self.avg_upload}>"
