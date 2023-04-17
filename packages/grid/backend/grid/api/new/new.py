# syft absolute
from syft.core.node.new.routes import make_routes

# grid absolute
from grid.core.node import worker

router = make_routes(worker=worker)
