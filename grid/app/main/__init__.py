import syft as sy
import torch as th
from flask import Blueprint

from .. import db, executor

# Avoid Pytorch deadlock issues
th.set_num_threads(1)

hook = sy.TorchHook(th)
local_worker = sy.VirtualWorker(hook, auto_add=False)
hook.local_worker.is_client_worker = False

main = Blueprint("main", __name__)
model_centric = Blueprint("model_centric", __name__)
data_centric = Blueprint("data_centric", __name__)
ws = Blueprint(r"ws", __name__)

from . import events, routes
from .dfl import auth
