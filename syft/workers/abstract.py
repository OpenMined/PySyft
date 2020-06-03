from abc import ABC
from abc import abstractmethod
from typing import Union

from syft.serde.syft_serializable import SyftSerializable


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

    @abstractmethod
    def send_obj(self, obj: object, location: "AbstractWorker"):
        """Send a torch object to a worker.

        Args:
            obj: An object to be sent.
            location: An AbstractWorker instance indicating the worker which should
                receive the object.
        """
        pass

    @abstractmethod
    def request_obj(
        self, obj_id: Union[str, int], location: "AbstractWorker", *args, **kwargs
    ) -> object:
        """Returns the requested object from specified location.

        Args:
            obj_id (int or string):  A string or integer id of an object to look up.
            location (BaseWorker): A BaseWorker instance that lets you provide the lookup
                location.
            user (object, optional): user credentials to perform user authentication.
            reason (string, optional): a description of why the data scientist wants to see it.
        Returns:
            A torch Tensor or Variable object.
        """
        pass
