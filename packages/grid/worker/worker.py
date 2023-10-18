# stdlib

# third party
from fastapi import FastAPI

# syft absolute
from syft.abstract_node import NodeSideType
from syft.abstract_node import NodeType
from syft.client.client import API_PATH
from syft.node.domain import Domain
from syft.node.enclave import Enclave
from syft.node.gateway import Gateway
from syft.node.node import get_dev_mode
from syft.node.node import get_enable_warnings
from syft.node.node import get_node_name
from syft.node.node import get_node_side_type
from syft.node.node import get_node_type
from syft.node.routes import make_routes
from syft.protocol.data_protocol import stage_protocol_changes

worker_classes = {
    NodeType.DOMAIN: Domain,
    NodeType.GATEWAY: Gateway,
    NodeType.ENCLAVE: Enclave,
}

node_name = get_node_name()
node_type = NodeType(get_node_type())
node_side_type = NodeSideType(get_node_side_type())
enable_warnings = get_enable_warnings()
if node_type not in worker_classes:
    raise NotImplementedError(f"node_type: {node_type} is not supported")
worker_class = worker_classes[node_type]
worker = worker_class(
    name=node_name,
    local_db=True,
    sqlite_path="./storage/",
    node_type=node_type,
    enable_warnings=enable_warnings,
    node_side_type=node_side_type,
)
router = make_routes(worker=worker)

app = FastAPI(title="Worker")

if get_dev_mode():
    print("Staging protocol changes...")
    status = stage_protocol_changes()
    print(status)


@app.get("/")
async def root() -> str:
    return f"OpenMined {node_type.value.capitalize()} Node Running"


app.include_router(router, prefix=API_PATH)
