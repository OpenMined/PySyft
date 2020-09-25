class GRID_EVENTS:
    JOIN = "join"
    FORWARD = "forward"
    MONITOR_ANSWER = "monitor-answer"


class NODE_EVENTS:
    MONITOR = "monitor"
    WEBRTC_SCOPE = "create-webrtc-scope"
    WEBRTC_OFFER = "webrtc-offer"
    WEBRTC_ANSWER = "webrtc-answer"


class USER_EVENTS(object):
    GET_ALL_USERS = "list-users"
    GET_SPECIFIC_USER = "list-user"
    SEARCH_USERS = "search-users"
    UPDATE_USER_EMAIL = "put-email"
    UPDATE_USER_PASSWORD = "put-password"
    UPDATE_USER_ROLE = "update-user-role"
    DELETE_USER = "delete-user"
    SIGNUP_USER = "signup-user"
    LOGIN_USER = "login-user"


class ROLE_EVENTS(object):
    CREATE_ROLE = "create-role"
    GET_SPECIFIC_ROLE = "list-role"
    GET_ALL_ROLES = "list-roles"
    PUT_ROLE = "put-role"
    DELETE_ROLE = "delete-role"


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
