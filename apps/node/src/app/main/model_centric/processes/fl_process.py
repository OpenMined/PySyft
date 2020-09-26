from ... import BaseModel, db


class FLProcess(BaseModel):
    """Federated Learning Process table.

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

    __tablename__ = "model_centric_fl_process"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255))
    version = db.Column(db.String(255))
    model = db.relationship("Model", backref="flprocess", uselist=False)
    averaging_plan = db.relationship("Plan", backref="avg_flprocess", uselist=False)
    plans = db.relationship("Plan", backref="plan_flprocess")
    protocols = db.relationship("Protocol", backref="protocol_flprocess")
    server_config = db.relationship("Config", backref="server_flprocess_config")
    client_config = db.relationship("Config", backref="client_flprocess_config")
    cycles = db.relationship("Cycle", backref="cycle_flprocess")

    def __str__(self):
        return f"<FederatedLearningProcess id : {self.id}>"
