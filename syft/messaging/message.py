import syft as sy

from syft.workers import AbstractWorker
from syft import codes


class Message:
    def __init__(self, msg_type, contents=None):
        self.msg_type = msg_type
        if contents is not None:
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

    def __str__(self):
        return f"({codes.code2MSGTYPE[self.msg_type]} {self.contents})"

    def __repr__(self):
        return self.__str__()


class CommandMessage(Message):
    def __init__(self, message, return_ids):
        super().__init__(codes.MSGTYPE.CMD)

        self.message = message
        self.return_ids = return_ids

    @property
    def contents(self):  # need this just because some methods assume the tuple form (legacy)
        return (self.message, self.return_ids)

    @staticmethod
    def simplify(ptr: "CommandMessage") -> tuple:
        """
        This function takes the attributes of a CommandMessage and saves them in a tuple
        Args:
            ptr (CommandMessage): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        # NOTE: we can skip calling _simplify on return_ids because they should already be
        # a list of simple types.
        return (ptr.msg_type, (sy.serde._simplify(ptr.message), ptr.return_ids))

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return CommandMessage(sy.serde._detail(worker, tensor_tuple[1][0]), tensor_tuple[1][1])


class ObjectMessage(Message):
    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.OBJ, contents)


class ObjectRequestMessage(Message):
    # TODO: add more efficieent detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.OBJ_REQ, contents)


class IsNoneMessage(Message):
    # TODO: add more efficieent detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.IS_NONE, contents)


class GetShapeMessage(Message):
    # TODO: add more efficieent detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.GET_SHAPE, contents)


class ForceObjectDeleteMessage(Message):
    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.FORCE_OBJ_DEL, contents)


class SearchMessage(Message):
    # TODO: add more efficieent detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.SEARCH, contents)
