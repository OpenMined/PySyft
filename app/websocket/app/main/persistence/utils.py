from .models import Worker as WorkerMDL
from .models import WorkerObject, TorchModel
from .models import db
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import torch
from syft.messaging.plan import Plan
import syft as sy
import os

# Cache keys already saved in database.
# Used to make snapshot more efficient.
last_snapshot_keys = set()


def set_database_config(app, test_config=None, verbose=False):
    """ Set configs to use SQL Alchemy library.

        Args:
            app: Flask application.
            test_config : Dictionary containing SQLAlchemy configs for test purposes.
            verbose : Level of flask application verbosity.
        Returns:
            app: Flask application.
        Raises:
            RuntimeError : If DATABASE_URL or test_config didn't initialized, RuntimeError exception will be raised.
    """
    global db
    db_url = os.environ.get("DATABASE_URL")
    migrate = Migrate(app, db)
    if test_config is None:
        if db_url:
            app.config.from_mapping(
                SQLALCHEMY_DATABASE_URI=db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False
            )
        else:
            raise RuntimeError(
                "Invalid database address : Set DATABASE_URL environment var or add test_config parameter at create_app method."
            )
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = test_config["SQLALCHEMY_DATABASE_URI"]
        app.config["TESTING"] = (
            test_config["TESTING"] if test_config.get("TESTING") else True
        )
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = (
            test_config["SQLALCHEMY_TRACK_MODIFICATIONS"]
            if test_config.get("SQLALCHEMY_TRACK_MODIFICATIONS")
            else False
        )
    app.config["VERBOSE"] = verbose
    db.init_app(app)
    return app


def snapshot(worker):
    """ Take a snapshot of worker's current state.

        Args:
            worker: Worker with objects that will be stored.
    """
    global last_snapshot_keys
    current_keys = set(worker._objects.keys())

    # Delete objects from database
    deleted_keys = last_snapshot_keys - current_keys
    for obj_key in deleted_keys:
        db.session.query(WorkerObject).filter_by(id=obj_key).delete()

    # Add new objects from database
    new_keys = current_keys - last_snapshot_keys

    objects = []
    for key in new_keys:
        obj = worker.get_obj(key)

        # If obj is an instance of Plan or TorchScriptModul we ignore it
        # We only store these objects when explictly sent using the Rest API.
        if isinstance(obj, (Plan, torch.jit.ScriptModule)):
            continue

        # If obj is a parameter we wrap the data and store it in the database
        # as an object
        elif isinstance(obj, torch.nn.Parameter):
            obj = obj.data.wrap()
            objects.append(WorkerObject(worker_id=worker.id, object=obj, id=key))
        # Otherwise we just store the object in the database
        else:
            objects.append(WorkerObject(worker_id=worker.id, object=obj, id=key))

    db.session.add_all(objects)
    db.session.commit()
    last_snapshot_keys = current_keys


def recover_objects(worker):
    """ Find or create a new worker.

        Args:
            worker: Local worker.
        Returns:
            Virtual worker filled by stored objects.
    """
    worker_mdl = WorkerMDL.query.filter_by(id=worker.id).first()
    if worker_mdl:
        global last_snapshot_keys

        # Recover objects
        objs = db.session.query(WorkerObject).filter_by(worker_id=worker.id).all()
        obj_dict = {}
        for obj in objs:
            obj_dict[obj.id] = obj.object

        # Recover models
        models = TorchModel.query.all()
        for result in models:
            model = sy.serde.deserialize(result.model)
            obj_dict[result.id] = model

        worker._objects = obj_dict
        last_snapshot_keys = set(obj_dict.keys())
    else:
        worker_mdl = WorkerMDL(id=worker.id)
        db.session.add(worker_mdl)
        db.session.commit()

    return worker
