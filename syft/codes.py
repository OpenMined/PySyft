class PLAN_CMDS(object):
    FETCH_PLAN = "fetch_plan"
    FETCH_PROTOCOL = "fetch_protocol"


class TENSOR_SERIALIZATION(object):
    TORCH = "torch"
    NUMPY = "numpy"
    TF = "tf"
    ALL = "all"


class GATEWAY_ENDPOINTS(object):
    SEARCH_TAGS = "/search"
    SEARCH_MODEL = "/search-model"
    SEARCH_ENCRYPTED_MODEL = "/search-encrypted-model"
    SELECT_MODEL_HOST = "/choose-model-host"
    SELECT_ENCRYPTED_MODEL_HOSTS = "/choose-encrypted-model-host"


class REQUEST_MSG(object):
    TYPE_FIELD = "type"
    GET_ID = "get-id"
    CONNECT_NODE = "connect-node"
    HOST_MODEL = "host-model"
    RUN_INFERENCE = "run-inference"
    LIST_MODELS = "list-models"
    DELETE_MODEL = "delete-model"
    RUN_INFERENCE = "run-inference"
    AUTHENTICATE = "authentication"


class RESPONSE_MSG(object):
    NODE_ID = "id"
    ERROR = "error"
    SUCCESS = "success"
    MODELS = "models"
    INFERENCE_RESULT = "prediction"
    SYFT_VERSION = "syft_version"
