from syft.grid.grid_client import connect
from syft.core.node.domain.domain import Domain
from syft.core.node.device.client import DeviceClient
from syft.grid.connections.http_connection import HTTPConnection

node = Domain(name="om-domain")
# node.immediate_services_with_reply.append(CreateWorkerService)
# node._register_services()  # re-register all services including SignalingService

try:
    node.private_device = connect(
        url="http://localhost:5000",  # Domain Address
        conn_type=HTTPConnection,  # HTTP Connection Protocol
        client_type=DeviceClient,
    )  # Device Client type
    node.in_memory_client_registry[node.private_device.device_id] = node.private_device
except:
    pass
