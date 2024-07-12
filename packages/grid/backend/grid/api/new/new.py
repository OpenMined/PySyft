# syft absolute
from syft.server.routes import make_routes

# server absolute
from grid.core.server import worker

router = make_routes(worker=worker)
