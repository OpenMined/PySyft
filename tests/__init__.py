GRID_NETWORK_PORT = "8000"
NETWORK_PORT = "3003"
NETWORK_URL = "http://localhost:{}".format(NETWORK_PORT)
NETWORK_WS_URL = "ws://localhost:{}".format(NETWORK_PORT)

PORTS = ["3000", "3001", "3002", "3003"]
IDS = ["Alice", "Bob", "Charlie", "Dan"]

worker_ports = {node_id: port for node_id, port in zip(IDS, PORTS)}
