from flask import Blueprint

import syft as sy
import torch as th

# Global variables must be initialized here.
hook = sy.TorchHook(th)
local_worker = sy.VirtualWorker(hook, auto_add=False)
hook.local_worker.is_client_worker = False

html = Blueprint(r"html", __name__)
ws = Blueprint(r"ws", __name__)


from . import routes, events
from .persistence.models import db
from . import auth
