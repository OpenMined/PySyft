import torch
import syft

from typing import List
from typing import Union

from syft.generic.frameworks.hook import hook_args


from syft.workers.abstract import AbstractWorker
from syft.generic.tensor import AbstractTensor


class PrivateTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags: set = None, description: str = None):
        """ Initialize a Private tensor, which manages permissions restricting get operations.

            Args:
                owner (BaseWorker, optional): A BaseWorker object to specify the worker on which
                the tensor is located.
                id (string or int, optional): An optional string or integer id of the PrivateTensor.
                tags (set, optional): A set of tags to label this tensor.
                description (string, optional): A brief description about this tensor.
        """
        super().__init__(tags, description)
        self.owner = owner
        self.id = id if id else syft.ID_PROVIDER.pop()
        self.child = None
        self.allowed_users = list()

    def allowed_to_get(self, user) -> bool:
        """ Overwrite native's allowed_to_get to verify if a specific user is allowed to get this tensor.
        
            Args:
                user (object): user to be verified.

            Returns:
                bool : A boolean value (True if the user is allowed and false if it isn't).
        """
        return user in self.allowed_users

    def register_credentials(self, users: Union[object, List[str]] = []) -> "PrivateTensor":
        """ Register a new user credential(s) into the list of allowed users to get this tensor.

            Args:
                users (object or List): Credential(s) to be registered.
        """
        if not hasattr(self, "allowed_users"):
            self.allowed_users = list()

        # If it's a List of credentials
        if isinstance(users, List):
            self.allowed_users += users
        else:
            self.allowed_users.append(users)

        return self

    @staticmethod
    def simplify(tensor: "PrivateTensor") -> tuple:
        """Takes the attributes of a PrivateTensor and saves them in a tuple.

        Args:
            tensor (PrivateTensor): a PrivateTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed private tensor.
        """

        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde._simplify(tensor.child)

        return (
            syft.serde._simplify(tensor.id),
            tensor.allowed_users,
            syft.serde._simplify(tensor.tags),
            syft.serde._simplify(tensor.description),
            chain,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PrivateTensor":
        """
            This function reconstructs a PrivateTensor given it's attributes in form of a tuple.
            Args:
                worker (AbstractWorker): the worker doing the deserialization
                tensor_tuple (tuple): a tuple holding the attributes of the PrivateTensor
            Returns:
                PrivateTensor: a PrivateTensor
            Examples:
                shared_tensor = detail(data)
        """

        tensor_id, allowed_users, tags, description, chain = tensor_tuple

        tensor = PrivateTensor(
            owner=worker,
            id=syft.serde._detail(worker, tensor_id),
            tags=syft.serde._detail(worker, tags),
            description=syft.serde._detail(worker, description),
        )

        tensor.allowed_users = [user.decode("utf-8") for user in allowed_users]

        if chain is not None:
            chain = syft.serde._detail(worker, chain)
            tensor.child = chain

        return tensor


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PrivateTensor)
