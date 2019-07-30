from flask import Blueprint

import syft as sy
import torch as th

hook = sy.TorchHook(th)

main = Blueprint("main", __name__)

from . import routes, events
