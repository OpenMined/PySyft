from abc import ABC

class Service(ABC):
    """Service is the interface that all services must implement in order to
    process messages in workers. As a note, a service should implement the
    processing for only one message.
    """
    def process(self, worker: "Worker", msg: "SyftMessage") -> "MesssageResponse":
        """Method to implement the processing logic of a message type.

        Args:
            worker (AbstractWorker): the worker on which themessage processing
                is being done.
            msg (SyftMessage): the message to be processed.

        Returns:
            MessageResponse: the response of the message processing.
        """
        pass

    def message_type_handler(self) -> type:
        """Method to return the type of the message that this service can
        handle.

        Returns:
            type: the type of the message processed by the Service.
        """
        pass
