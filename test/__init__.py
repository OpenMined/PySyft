GATEWAY_PORT = "3004"
GATEWAY_URL = "http://localhost:{}".format(GATEWAY_PORT)

PORTS = ["3000", "3001", "3002"]
IDS = ["bob", "alice", "james"]

worker_ports = {node_id: port for node_id, port in zip(IDS, PORTS)}
