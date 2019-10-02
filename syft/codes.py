class MSGTYPE(object):
    CMD = 1
    OBJ = 2
    OBJ_REQ = 3
    OBJ_DEL = 4
    EXCEPTION = 5
    IS_NONE = 6
    GET_SHAPE = 7
    SEARCH = 8
    FORCE_OBJ_DEL = 9
    PLAN_CMD = 10


class PLAN_CMDS(object):
    FETCH_PLAN = "fetch_plan"
    FETCH_PROTOCOL = "fetch_protocol"


# Build automatically the reverse map from codes to messaging types
code2MSGTYPE = {}
for code in dir(MSGTYPE):
    if "__" not in code:
        key = getattr(MSGTYPE, code)
        code2MSGTYPE[key] = code
