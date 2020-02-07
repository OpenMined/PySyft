class GRID_MSG(object):
    TYPE_FIELD = "type"
    DATA_FIELD = "data"
    SOCKET_PING = "socket-ping"
    GET_PROTOCOL = "get-protocol"
    PEER_LEFT = "webrtc: peer-left"
    INTERNAL_MSG = "webrtc: internal-message"
    JOIN_ROOM = "webrtc: join-room"


class RESPONSE_MSG(object):
    PROTOCOL_ID = "protocolId"
    ASSIGNMENT = "assignment"
    WORKER_ID = "workerId"
    SCOPE_ID = "scopeId"
    ALIVE = "alive"
    ROLE = "role"
    PLAN = "plan"
