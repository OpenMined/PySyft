# stdlib
import os

# third party
from fastapi import FastAPI
from new_routes import make_routes

# syft absolute
from syft.core.node.new.client import API_PATH
from syft.core.node.worker import Worker

node_name = os.environ.get("NODE_NAME", "default_node_name")
worker = Worker.named(name=node_name, local_db=True, sqlite_path="/storage/")
print("Worker signing key", worker.signing_key)
router = make_routes(worker=worker)

app = FastAPI(title="Worker")
app.include_router(router, prefix=API_PATH)
