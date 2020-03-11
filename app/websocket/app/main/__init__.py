from flask import Blueprint

import syft as sy
import torch as th

from typing import List
from typing import Tuple
from typing import Union

from syft.serde import serialize, deserialize
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.pointers.pointer_tensor import PointerTensor

# Global variables must be initialized here.
hook = sy.TorchHook(th)
local_worker = sy.VirtualWorker(hook, auto_add=False)
hook.local_worker.is_client_worker = False

html = Blueprint(r"html", __name__)
ws = Blueprint(r"ws", __name__)


from . import routes, events
from . import auth
