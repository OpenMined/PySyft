from .models import Worker as WorkerMDL
from .models import WorkerObject
from .models import db

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
    objects = [
        WorkerObject(worker_id=worker.id, object=worker._objects[key], id=key)
        for key in new_keys
    ]

    result = db.session.add_all(objects)
    db.session.commit()
    last_snapshot_keys = current_keys


def recover_objects(hook):
    """ Find or create a new worker.
        
        Args:
            hook : Global hook.
        Returns:
            worker : Virtual worker (filled by stored objects) that will replace hook.local_worker.
    """
    worker = hook.local_worker
    worker_mdl = WorkerMDL.query.filter_by(id=worker.id).first()
    if worker_mdl:
        global last_snapshot_keys
        objs = db.session.query(WorkerObject).filter_by(worker_id=worker.id).all()
        obj_dict = {}
        for obj in objs:
            obj_dict[obj.id] = obj.object
        worker._objects = obj_dict
        last_snapshot_keys = set(obj_dict.keys())
    else:
        worker_mdl = WorkerMDL(id=worker.id)
        db.session.add(worker_mdl)
        db.session.commit()
    return worker
