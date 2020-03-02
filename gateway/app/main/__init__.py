from flask import Blueprint
import syft as sy
import torch as th

hook = sy.TorchHook(th)

main = Blueprint("main", __name__)
ws = Blueprint(r"ws", __name__)

from .storage.models import db

from . import routes, events
