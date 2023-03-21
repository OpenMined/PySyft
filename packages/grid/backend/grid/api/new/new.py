# grid absolute
from grid.core.node import worker

# relative
from .new_routes import make_routes

router = make_routes(worker=worker)
