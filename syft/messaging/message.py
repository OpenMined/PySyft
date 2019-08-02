import syft as sy

from syft.workers import AbstractWorker


class Message:
    def __init__(self, msg_type, contents):
        self.msg_type = msg_type
        self.contents = contents

    def _simplify(self):
        return (self.msg_type, self.contents)

    @staticmethod
    def simplify(ptr: "Message") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple
        Args:
            ptr (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """

        return (ptr.msg_type, sy.serde._simplify(ptr.contents))

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return Message(tensor_tuple[0], sy.serde._detail(worker, tensor_tuple[1]))
