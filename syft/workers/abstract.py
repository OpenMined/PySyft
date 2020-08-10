from abc import ABC
from abc import abstractmethod
from syft.generic.abstract.syft_serializable import SyftSerializable


class AbstractWorker(ABC, SyftSerializable):
    @abstractmethod
    def _send_msg(self, message: bin, location: "AbstractWorker"):
        """Sends message from one worker to another.

        As AbstractWorker implies, you should never instantiate this class by
        itself. Instead, you should extend AbstractWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: A binary message to be sent from one worker
                to another.
            location: A AbstractWorker instance that lets you provide the
                destination to send the message.
        """
        pass

    @abstractmethod
    def _recv_msg(self, message: bin):
        """Receives the message.

        As AbstractWorker implies, you should never instantiate this class by
        itself. Instead, you should extend AbstractWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: The binary message being received.
        """
        pass
