class MSG_FIELD:
    TYPE = "type"
    DATA = "data"
    WORKER_ID = "worker_id"
    MODEL = "model"
    MODEL_ID = "model_id"
    ALIVE = "alive"


class CONTROL_EVENTS(object):
    SOCKET_PING = "socket-ping"


class WEBRTC_EVENTS(object):
    PEER_LEFT = "webrtc: peer-left"
    INTERNAL_MSG = "webrtc: internal-message"
    JOIN_ROOM = "webrtc: join-room"


class FL_EVENTS(object):
    REPORT = "federated/report"
    AUTHENTICATE = "federated/authenticate"
    CYCLE_REQUEST = "federated/cycle-request"


class CYCLE(object):
    STATUS = "status"
    KEY = "request_key"
    PING = "ping"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    VERSION = "version"
    PLANS = "plans"
    PROTOCOLS = "protocols"
    CLIENT_CONFIG = "client_config"
    SERVER_CONFIG = "server_config"
    TIMEOUT = "timeout"
    DIFF = "diff"
    AVG_PLAN = "averaging_plan"


class RESPONSE_MSG(object):
    ERROR = "error"
