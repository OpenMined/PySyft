class PLAN_CMDS(object):  # noqa: N801
    FETCH_PLAN = "fetch_plan"
    FETCH_PROTOCOL = "fetch_protocol"


class TENSOR_SERIALIZATION(object):  # noqa: N801
    TORCH = "torch"
    NUMPY = "numpy"
    TF = "tf"
    ALL = "all"


class GATEWAY_ENDPOINTS(object):  # noqa: N801
    SEARCH_TAGS = "/search"
    SEARCH_MODEL = "/search-model"
    SEARCH_ENCRYPTED_MODEL = "/search-encrypted-model"
    SELECT_MODEL_HOST = "/choose-model-host"
    SELECT_ENCRYPTED_MODEL_HOSTS = "/choose-encrypted-model-host"


class REQUEST_MSG(object):  # noqa: N801
    TYPE_FIELD = "type"
    GET_ID = "get-id"
    CONNECT_NODE = "connect-node"
    HOST_MODEL = "host-model"
    RUN_INFERENCE = "run-inference"
    LIST_MODELS = "list-models"
    DELETE_MODEL = "delete-model"
    RUN_INFERENCE = "run-inference"
    AUTHENTICATE = "authentication"


class RESPONSE_MSG(object):  # noqa: N801
    NODE_ID = "id"
    ERROR = "error"
    SUCCESS = "success"
    MODELS = "models"
    INFERENCE_RESULT = "prediction"
    SYFT_VERSION = "syft_version"


class MSG_FIELD:
    TYPE = "type"
    FROM = "from"
    DESTINATION = "destination"
    CONTENT = "content"
    NODE_ID = "node_id"
    PAYLOAD = "payload"
    NODES = "nodes"
    MODELS = "models"
    DATASETS = "datasets"
    CPU = "cpu"
    MEM_USAGE = "mem_usage"


class NODE_EVENTS:
    MONITOR = "monitor"
    WEBRTC_SCOPE = "create-webrtc-scope"
    WEBRTC_OFFER = "webrtc-offer"
    WEBRTC_ANSWER = "webrtc-answer"


class GRID_EVENTS:
    JOIN = "join"
    FORWARD = "grid-forward"
    FORWARD = "forward"
    MONITOR_ANSWER = "monitor-answer"
