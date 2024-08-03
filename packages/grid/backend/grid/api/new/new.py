# syft absolute
# server absolute
from grid.core.server import worker
from syft.server.routes import make_routes

router = make_routes(worker=worker)
