import datetime
import json
from flask_sqlalchemy import SQLAlchemy
import uuid
from .. import db

import syft as sy


class Model(db.Model):
    """ Model table that represents the AI Models.
        Columns:
            id (Int, Primary Key) : Model's id, used to recover stored model.
            version (String) : Model version.
            checkpoints (ModelCheckPoint) : Model Checkpoints. (One to Many relationship)
            fl_process_id (Integer, ForeignKey) : FLProcess Foreign Key.
    """

    __tablename__ = "__model__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    version = db.Column(db.String())
    checkpoints = db.relationship("ModelCheckPoint", backref="model")
    fl_process_id = db.Column(
        db.BigInteger, db.ForeignKey("__fl_process__.id"), unique=True
    )

    def __str__(self):
        return f"<Model  id: {self.id}, version: {self.version}>"


class ModelCheckPoint(db.Model):
    """ Model's save points.
        Columns:
            id (Integer, Primary Key): Checkpoint ID.
            values (Binary): Value of the model at a given checkpoint.
            model_id (String, Foreign Key): Model's ID.
    """

    __tablename__ = "__model_checkpoint__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    values = db.Column(db.LargeBinary)
    alias = db.Column(db.String)
    model_id = db.Column(db.String, db.ForeignKey("__model__.id"))

    @property
    def object(self):
        return sy.serde.deserialize(self.values)

    @object.setter
    def object(self):
        self.data = sy.serde.serialize(self.values)

    def __str__(self):
        return f"<CheckPoint id: {self.id}, model_id: {self.model_id}>"


class Plan(db.Model):
    """ Plan table that represents Syft Plans.
        Columns:
            id (Integer, Primary Key): Plan ID.
            name (String): Plan name.
            value (String): String  (List of operations)
            value_ts (String): String (TorchScript)
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
            f"<Plan id: {self.id}, values: {self.value}, torchscript: {self.value_ts}>"
        )


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


class Config(db.Model):
    """ Configs table.
        Columns:
            id (Integer, Primary Key): Config ID.
            config (String): Dictionary
            is_server_config (Boolean) : Boolean flag to indicate if it is a server config (True) or client config (False)
            fl_process_id (Integer, Foreign Key) : Referece to FL Process.
    """

    __tablename__ = "__config__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    config = db.Column(db.PickleType)
    is_server_config = db.Column(db.Boolean, default=False)
    fl_process_id = db.Column(db.BigInteger, db.ForeignKey("__fl_process__.id"))

    def __str__(self):
        return f"<Config id: {self.id} , configs: {self.config}>"


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


class FLProcess(db.Model):
    """ Federated Learning Process table.
        Columns:
            id (Integer, Primary Key): Federated Learning Process ID.
            name (String) : Federated Process name.
            version (String) : FL Process version.
            model (Model): Model.
            averaging_plan (Plan): Averaging Plan.
            plans: Generic Plans (such as training plan and validation plan)
            protocols (Protocol) : Generic protocols.
            server_config (Config): Server Configs.
            client_config (Config): Client Configs.
            cycles [Cycles]: FL Cycles.
    """

    __tablename__ = "__fl_process__"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String())
    version = db.Column(db.String())
    model = db.relationship("Model", backref="flprocess", uselist=False)
    averaging_plan = db.relationship("Plan", backref="avg_flprocess", uselist=False)
    plans = db.relationship("Plan", backref="plan_flprocess")
    protocols = db.relationship("Protocol", backref="protocol_flprocess")
    server_config = db.relationship("Config", backref="server_flprocess_config")
    client_config = db.relationship("Config", backref="client_flprocess_config")
    cycles = db.relationship("Cycle", backref="cycle_flprocess")

    def __str__(self):
        return f"<FederatedLearningProcess id : {self.id}>"


class GridNodes(db.Model):
    """ Grid Nodes table that represents connected grid nodes.
    
        Columns:
            id (primary_key) : node id, used to recover stored grid nodes (UNIQUE).
            address: Address of grid node.
    """

    __tablename__ = "__gridnode__"

    id = db.Column(db.String(), primary_key=True)
    address = db.Column(db.String())

    def __str__(self):
        return f"< Grid Node {self.id} : {self.address}>"
