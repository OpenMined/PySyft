from flask import Blueprint
from flask import session, request
from .models import Worker as WorkerMDL
from .models import WorkerObject
from .models import db
import binascii
import torch as th
import syft as sy

hook = sy.TorchHook(th)

worker_router_bp = Blueprint("grid_worker", __name__)
worker_router_bp.config = {}


@worker_router_bp.record
def record_params(setup_state):
    """
    Copy config so we can easily work with it
    """
    app = setup_state.app
    worker_router_bp.config = dict(
        [(key, value) for (key, value) in app.config.items()]
    )


def _store_worker(worker, worker_mdl: WorkerMDL, worker_name: str = "worker"):
    """
    Persist workers to our persistence layer
    """
    db.session.query(WorkerObject).filter_by(worker_id=worker_mdl.id).delete()
    objects = [
        WorkerObject(worker_id=worker_mdl.id, object=obj)
        for key, obj in worker._objects.items()
    ]
    result = db.session.add_all(objects)
    db.session.commit()


def _maybe_create_worker(
    worker_name: str = "worker", virtual_worker_id: str = "grid", verbose: bool = False
):
    """
    Find or create a worker by public_id

    """
    worker_mdl = WorkerMDL.query.filter_by(public_id=worker_name).first()
    if worker_mdl is None:
        worker_mdl = WorkerMDL(public_id=worker_name)
        db.session.add(worker_mdl)
        db.session.commit()
        worker = sy.VirtualWorker(hook, virtual_worker_id, auto_add=False)
        if verbose:
            print("\t \nCREATING NEW WORKER!!")
    else:
        worker = sy.VirtualWorker(hook, virtual_worker_id, auto_add=False)
        for obj in worker_mdl.worker_objects:
            worker.register_obj(obj.object)
        if verbose:
            print("\t \nFOUND OLD WORKER!! " + str(worker._objects.keys()))
    return worker, worker_mdl


def _request_message(worker):
    """
    Transform HTTP message into an action that can be invoked by a worker
    """
    message = request.form["message"]
    message = binascii.unhexlify(message[2:-1])
    response = worker._recv_msg(message)
    response = str(binascii.hexlify(response))
    return response


@worker_router_bp.route("/")
def success():
    """
    Entrypoint
    """
    return "success"


@worker_router_bp.route("/identity/")
def is_this_an_opengrid_node():
    """This exists because in the automation scripts which deploy nodes,
    there's an edge case where the 'node already exists' but sometimes it
    can be an app that does something totally different. So we want to have
    some endpoint which just casually identifies this server as an OpenGrid
    server."""
    return "OpenGrid"


@worker_router_bp.route("/cmd/", methods=["POST"])
def cmd():
    """
    Worker command execution endpoint
    """
    verbose = worker_router_bp.config["VERBOSE"]
    try:
        worker, worker_mdl = _maybe_create_worker("worker", "grid")
        worker.verbose = verbose
        sy.torch.hook.local_worker.add_worker(worker)
        response = _request_message(worker)
        if verbose:
            print("\t NEW WORKER STATE:" + str(worker._objects.keys()) + "\n\n")
        _store_worker(worker, worker_mdl, "worker")
        db.session.flush()
        return response
    except Exception as e:
        return str(e)
