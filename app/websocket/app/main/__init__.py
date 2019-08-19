from flask import Blueprint

import syft as sy
import torch as th

# Global variables must be initialized here.

hook = sy.TorchHook(th)

main = Blueprint("main", __name__)

from . import routes, events
