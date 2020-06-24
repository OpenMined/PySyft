from typing import List

import syft

from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.abstract.sendable import AbstractSendable

# ATTENTION!
# Currently, this class is not used but we might start using it when PyTorch introduces support
# to convert from OnnxToPytorch


class OnnxModel(AbstractSendable):
    def __init__(
        self,
        serialized_model: bytes,
        id: int = None,
        owner: "syft.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        super(OnnxModel, self).__init__(id, owner, tags, description, None)
        self.serialized_model = serialized_model

    def get_class_attributes(self):
        return {"serialized_model": self.serialized_model}

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        **kwargs,
    ) -> ObjectPointer:
        """Creates a pointer to the "self" OnnxModel object.

        Returns:
            An ObjectPointer pointer to self.
        """

        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location is not None and location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = syft.ID_PROVIDER.pop()

        ptr = ObjectPointer.create_pointer(
            self, location, id_at_location, register, owner, ptr_id, garbage_collect_data, **kwargs
        )

        return ptr

    @staticmethod
    def simplify(worker: AbstractWorker, model: "OnnxModel") -> tuple:
        """
        Takes the attributes of a FixedPrecisionTensor and saves them in a tuple.

        Args:
            worker: the worker doing the serialization
            model: an OnnxModel.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed precision tensor.
        """
        return (
            syft.serde.msgpack.serde._simplify(worker, model.serialized_model),
            syft.serde.msgpack.serde._simplify(worker, model.id),
            syft.serde.msgpack.serde._simplify(worker, model.tags),
            syft.serde.msgpack.serde._simplify(worker, model.description),
        )
