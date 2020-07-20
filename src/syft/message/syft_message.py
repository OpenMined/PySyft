from ...common.message import AbstractMessage
from ...common.id import UID


class CommunicationEnvelop(object):
    """
    A syft message contains information about who should the message
    be communicated with next. This information includes:
    - As a worker receiving this message, should I forward this message?
        if so to who (specific routes or broadcast)
        and how (on what protocols)?
    - As a worker receiving this msg, should I sign the message before forwarding?
           eg. if the user is keeping count on the number of forwards.
    - As a worker receiving this message, If this is a processing message,
        and I need to inform some party when I'm done processing,
        who should I inform and how?
    - As a worker sending this message, I should attach this object
        to my message.

    All the above assumes that worker of this message has visiblity to the workers
    its been asked to communicate with. If it has no visibility it'll log a warning
    and
    """
    def __init__(self, should_forward: Boolean = False):
        self.should_forward = should_forward
        self.should_publish_outcomes = should_publish_outcomes
        self.forward_routes = []
        self.forward_broadcast = None
        self.outcomes_routes = []
        self.outcomes_broadcast = []

    def is_valid(self):
        """
        Ensure the following is met:
          - if should_forward is True > forward routes or broadcast should be
            defined.
          - if should_publish_outcomes is True > outcomes routes or broadcast
            should be defined.
        """
        return True


class SyftMessage(AbstractMessage):
    def __init__(self, route: Route, comms: CommunicationEnvelop, msg_id: UID = None) -> None:
        self.route = route
        self.msg_id = msg_id
        self.communication = comms

    def _forward(self):
        if not self.communication.should_forward:
            # log forwarding hasn't been requested.
            return
        if not self.communciation.forward_routes or self.communication.forward_broadcast:
            # log no forward routes or broadcast were defined.
            return

    def _publish_outcomes(self, outcomes = []):
        if not self.communication.should_publish_outcomes:
            # log outcomes publishing has been requested.
            return

    def _process_internals(self):
        pass

    def process(self):
        """
        This should be implemented by specific message types.
        """
        self._process_internals()
