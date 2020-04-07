from .. import db


class Worker(db.Model):
    """ Web / Mobile worker table.
        Columns:
            id (String, Primary Key): Worker's ID.
            format_preference (String): either "list" or "ts"
            ping (Int): Ping rate.
            avg_download (Int): Download rate.
            avg_upload (Int): Upload rate.
            worker_cycles (WorkerCycle): Relationship between workers and cycles (One to many).
    """

    __tablename__ = "__worker__"

    id = db.Column(db.String, primary_key=True)
    format_preference = db.Column(db.String())
    ping = db.Column(db.BigInteger)
    avg_download = db.Column(db.BigInteger)
    avg_upload = db.Column(db.BigInteger)
    worker_cycle = db.relationship("WorkerCycle", backref="worker")

    def __str__(self):
        return f"<Worker id: {self.id}, format_preference: {self.format_preference}, ping : {self.ping}, download: {self.download}, upload: {self.upload}>"
