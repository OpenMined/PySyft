import torch
import syft
from typing import List

from syft.workers.abstract import AbstractWorker
from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.overload import overloaded


class PrivateTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags: set = None, description: str = None):
        super().__init__(tags, description)
        self.owner = owner
        self.id = id if id else syft.ID_PROVIDER.pop()
        self.child = None
        self.allowed_users = list()

    def allowed_to_get(self, user) -> bool:
        return user in self.allowed_users

    def register_users(self, users: List[str] = []) -> "PrivateTensor":
        if not hasattr(self, "allowed_users"):
            self.allowed_users = list()
        self.allowed_users += users
        return self

    def add_new_user(self, user: str):
        self.allowed_users.append(user)

    @overloaded.method
    def t(self, _self, *args, **kwargs):
        """Transpose a tensor. Hooked is handled by the decorator"""
        response = getattr(_self, "t")(*args, **kwargs)

        return response

    @staticmethod
    def simplify(tensor: "PrivateTensor") -> tuple:
        """Takes the attributes of a PrivateTensor and saves them in a tuple.

        Args:
            tensor: a PrivateTensor.

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
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the PrivateTensor
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
