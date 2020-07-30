from syft.common.message import AbstractMessage
from syft.common.id import UID
from syft.core.io.route import Route

class SyftBroadcastMessage(AbstractMessage):
    def to_dict(self):
        """
        for testing, when protobuf is unavailable.
        """
        res = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                value = value.__dict__
            res.update({key: value})
        res['class'] = self.__class__.__name__
        return res

class BroadcastMessageSpecs(object):
    """
    Placeholder for schema'ed messages.
    """
    def __init__(self, specs: dict):
        for key, value in specs.items():
            self.__dict__[key] = value


class SyftBroadcastMessageWithoutReply(SyftBroadcastMessage):
    def __init__(self, specs: BroadcastMessageSpecs, msg_id: UID = None) -> None:
        self.specs = specs
        self.msg_id = msg_id

class SyftBroadcastMessageWithReply(SyftBroadcastMessage):
    def __init__(self, specs: BroadcastMessageSpecs, reply_on_route: Route,
        msg_id: UID = None) -> None:
        self.specs = specs
        self.msg_id = msg_id
        self.reply_on_route = reply_on_route
