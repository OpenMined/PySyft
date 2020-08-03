import syft as sy
import torch as th
from flask import Blueprint

from .. import db, BaseModel, executor

# Avoid Pytorch deadlock issues
th.set_num_threads(1)

hook = sy.TorchHook(th)
local_worker = sy.VirtualWorker(hook, auto_add=False)
hook.local_worker.is_client_worker = False

main_routes = Blueprint("main", __name__)
model_centric_routes = Blueprint("model-centric", __name__)
data_centric_routes = Blueprint("data-centric", __name__)
ws = Blueprint(r"ws", __name__)

from . import events, routes
from .data_centric import auth
