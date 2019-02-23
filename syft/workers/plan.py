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
        (some_type, (msg_type, contents)) = serde.decompress(bin_message)

        self.plan.append(bin_message)
        self.readable_plan.append((msg_type, contents))
        
        # we can't receive the results of a plan without
        # executing it. So, execute the plan.
        if msg_type == MSGTYPE.OBJ_REQ:
            return self.execute_plan()

        return serde.serialize(None)

    def execute_plan(self):

        print("Execute Plan")
        response = None
        for bin_message, message in zip(self.plan, self.readable_plan):
            print(message)
            response = self.recv_msg(bin_message)

        self.plan = []
        self.readable_plan = []
        return response