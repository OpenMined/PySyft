from flask import Blueprint

import syft as sy
import torch as th

# Avoid Pytorch deadlock issues
th.set_num_threads(1)

hook = sy.TorchHook(th)
local_worker = sy.VirtualWorker(hook, auto_add=False)
hook.local_worker.is_client_worker = False

main = Blueprint("main", __name__)
ws = Blueprint(r"ws", __name__)


from .. import db, executor
from .dfl import auth
from . import routes, events
