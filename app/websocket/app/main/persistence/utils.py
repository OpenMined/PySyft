from .models import Worker as WorkerMDL
from .models import WorkerObject, TorchModel
from .models import db

import torch
from syft.messaging.plan import Plan
import syft as sy


# Cache keys already saved in database.
# Used to make snapshot more efficient.
last_snapshot_keys = set()


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

    objects, models = [], []
    for key in new_keys:
        obj = worker.get_obj(key)

        # If obj is a Plan we save in the database as a model
        if isinstance(obj, Plan):
            models.append(TorchModel(id=key, model=sy.serde.serialize(obj)))
        # If obj is Jit we ignore it
        elif isinstance(obj, torch.jit.ScriptModule):
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
    db.session.add_all(models)
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
