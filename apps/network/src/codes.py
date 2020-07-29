class GRID_EVENTS:
    JOIN = "join"
    FORWARD = "forward"
    MONITOR_ANSWER = "monitor-answer"


class NODE_EVENTS:
    MONITOR = "monitor"
    WEBRTC_SCOPE = "create-webrtc-scope"
    WEBRTC_OFFER = "webrtc-offer"
    WEBRTC_ANSWER = "webrtc-answer"


class MSG_FIELD:
    TYPE = "type"
    FROM = "from"
    DESTINATION = "destination"
    CONTENT = "content"
    NODE_ID = "node_id"
    MODELS = "models"
    DATASETS = "datasets"
    NODES = "nodes"
    STATUS = "status"
    SUCCESS = "success"
    ERROR = "error"
    CPU = "cpu"
    MEM_USAGE = "mem_usage"


class WORKER_PROPERTIES:
    HEALTH_CHECK_INTERVAL = 15
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
    PING_THRESHOLD = 100
