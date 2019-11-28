class PLAN_CMDS(object):
    FETCH_PLAN = "fetch_plan"
    FETCH_PROTOCOL = "fetch_protocol"


class TENSOR_SERIALIZATION(object):
    TORCH = "torch"
    NUMPY = "numpy"
    TF = "tf"
    ALL = "all"


class REQUEST_MSG(object):
    TYPE_FIELD = "type"
    GET_ID = "get-id"
    CONNECT_NODE = "connect-node"
    HOST_MODEL = "host-model"
    RUN_INFERENCE = "run-inference"
    LIST_MODELS = "list-models"
    DELETE_MODEL = "delete-model"
    RUN_INFERENCE = "run-inference"


class RESPONSE_MSG(object):
    NODE_ID = "id"
    ERROR = "error"
    SUCCESS = "success"
    MODELS = "models"
    INFERENCE_RESULT = "prediction"
