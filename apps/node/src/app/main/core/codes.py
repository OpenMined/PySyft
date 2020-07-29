class MSG_FIELD:
    TYPE = "type"
    DATA = "data"
    WORKER_ID = "worker_id"
    MODEL = "model"
    MODEL_ID = "model_id"
    ALIVE = "alive"
    ALLOW_DOWNLOAD = "allow_download"
    ALLOW_REMOTE_INFERENCE = "allow_remote_inference"
    MPC = "mpc"
    PROPERTIES = "model_properties"
    SIZE = "model_size"
    SYFT_VERSION = "syft_version"
    REQUIRES_SPEED_TEST = "requires_speed_test"
    USERNAME_FIELD = "username"
    PASSWORD_FIELD = "password"


class CONTROL_EVENTS(object):
    SOCKET_PING = "socket-ping"


class WEBRTC_EVENTS(object):
    PEER_LEFT = "webrtc: peer-left"
    INTERNAL_MSG = "webrtc: internal-message"
    JOIN_ROOM = "webrtc: join-room"


class MODEL_CENTRIC_FL_EVENTS(object):
    HOST_FL_TRAINING = "model-centric/host-training"
    REPORT = "model-centric/report"
    AUTHENTICATE = "model-centric/authenticate"
    CYCLE_REQUEST = "model-centric/cycle-request"


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
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RESPONSE_MSG(object):
    ERROR = "error"
    SUCCESS = "success"
