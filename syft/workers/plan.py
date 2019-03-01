from .base import BaseWorker
from syft.codes import MSGTYPE
from syft import serde


class Plan(BaseWorker):
    """This worker does not send messages or execute any commands. Instead,
    it simply records messages that are sent to it such that message batches
    (called 'Plans') can be created and sent once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = list()
        self.readable_plan = list()

    def _send_msg(self, message, location):
        return location._recv_msg(message)

    def _recv_msg(self, bin_message):
        (some_type, (msg_type, contents)) = serde.deserialize(bin_message, detail=False)

        if msg_type != MSGTYPE.OBJ:
            self.plan.append(bin_message)
            self.readable_plan.append((some_type, (msg_type, contents)))

        # we can't receive the results of a plan without
        # executing it. So, execute the plan.
        if msg_type in (MSGTYPE.OBJ_REQ, MSGTYPE.IS_NONE, MSGTYPE.GET_SHAPE):
            return self.execute_plan()

        return serde.serialize(None)

    def execute_plan(self, on_worker=None):

        if on_worker is None:
            on_worker = self

        print("Execute Plan")
        response = None
        for bin_message, message in zip(self.plan, self.readable_plan):
            print(message)
            bin_message = serde.serialize(message, simplified=True)
            response = on_worker.recv_msg(bin_message)

        self.plan = []
        self.readable_plan = []
        return response
