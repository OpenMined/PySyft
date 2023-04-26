# syft absolute
from syft.node.routes import make_routes

# grid absolute
from grid.core.node import worker

router = make_routes(worker=worker)
