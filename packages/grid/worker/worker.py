# stdlib
import os

# third party
from fastapi import FastAPI

# syft absolute
from syft.abstract_node import NodeType
from syft.client.client import API_PATH
from syft.node.domain import Domain
from syft.node.enclave import Enclave
from syft.node.gateway import Gateway
from syft.node.routes import make_routes

worker_classes = {
    NodeType.DOMAIN: Domain,
    NodeType.GATEWAY: Gateway,
    NodeType.ENCLAVE: Enclave,
}

node_name = os.environ.get("NODE_NAME", "default_node_name")
node_type = NodeType(os.environ.get("NODE_TYPE", "domain"))
if node_type not in worker_classes:
    raise NotImplementedError(f"node_type: {node_type} is not supported")
worker_class = worker_classes[node_type]
worker = worker_class(
    name=node_name, local_db=True, sqlite_path="/storage/", node_type=node_type
)
router = make_routes(worker=worker)

app = FastAPI(title="Worker")


@app.get("/")
async def root() -> str:
    return f"OpenMined {node_type.value.capitalize()} Node Running"


app.include_router(router, prefix=API_PATH)
