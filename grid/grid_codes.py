class REQUEST_MSG(object):
    TYPE_FIELD = "type"
    GET_ID = "get-id"
    CONNECT_NODE = "connect-node"
    AUTHENTICATE = "authentication"
    HOST_MODEL = "host-model"
    RUN_INFERENCE = "run-inference"
    LIST_MODELS = "list-models"
    DELETE_MODEL = "delete-model"
    DOWNLOAD_MODEL = "download-model"
    SYFT_COMMAND = "syft-command"
    PING = "socket-ping"


class RESPONSE_MSG(object):
    NODE_ID = "id"
    INFERENCE_RESULT = "prediction"
    MODELS = "models"
    ERROR = "error"
    SUCCESS = "success"
