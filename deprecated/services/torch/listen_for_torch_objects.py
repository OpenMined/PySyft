from ... import channels
from ..base import BaseService


class ListenForTorchObjectsService(BaseService):

    # this service listens to a channel specifically made for it to receive messages containing
    # torch objects and commands

    def __init__(self, worker):
        super().__init__(worker)

        self.worker = worker

        def print_messages(message):
            message = self.worker.decode_message(message)
            print(message)

        listen_for_callback_channel = channels.torch_listen_for_obj_callback(
            self.worker.id)
        self.worker.listen_to_channel(listen_for_callback_channel,
                                      print_messages)
