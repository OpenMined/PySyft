class MSGTYPE(object):
    CMD = 1
    OBJ = 2
    OBJ_REQ = 3
    OBJ_DEL = 4
    EXCEPTION = 5
    IS_NONE = 6


code2MSGTYPE = {}
code2MSGTYPE[1] = "CMD"
code2MSGTYPE[2] = "OBJ"
code2MSGTYPE[3] = "OBJ_REQ"
code2MSGTYPE[4] = "OBJ_DEL"
code2MSGTYPE[5] = "EXCEPTION"
code2MSGTYPE[6] = "IS_NONE"
