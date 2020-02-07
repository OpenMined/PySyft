GATEWAY_PORT = "3004"
GATEWAY_URL = "http://localhost:{}".format(GATEWAY_PORT)
GATEWAY_WS_URL = "ws://localhost:{}".format(GATEWAY_PORT)

PORTS = ["3000", "3001", "3002", "3003"]
IDS = ["bob", "alice", "james", "dan"]

worker_ports = {node_id: port for node_id, port in zip(IDS, PORTS)}
