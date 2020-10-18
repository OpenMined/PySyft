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
