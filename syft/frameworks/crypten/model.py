from typing import List

import torch

import syft

from syft.frameworks.crypten import utils
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

from syft_proto.frameworks.crypten.onnx_model_pb2 import OnnxModel as OnnxModelPB

"""
TODO: When the problem converting Onnx serialized models to PyTorch is solved
we can have serialized models on a SINGLE worker.

This model will be known by a single party and when the party will call
the crypten.load function it will deserialize the model to a PyTorch one
and share it with the other parties.

In this scenario, the worker that started the computation will not know about the
instrinsics of the architecture and *only one* worker can have knowledge about the
model
"""


class OnnxModel(AbstractSendable):
    def __init__(
        self,
        serialized_model: bytes = None,
        id: int = None,
        owner: "syft.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        super(OnnxModel, self).__init__(id, owner, tags, description, None)
        self.serialized_model = serialized_model

    @classmethod
    def fromModel(
        cls,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        id: int = None,
        owner: "syft.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        serialized_model = utils.pytorch_to_onnx(model, dummy_input)

        return cls(serialized_model, id, owner, tags, description)

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

    def to_crypten(self):
        return utils.onnx_to_crypten(self.serialized_model)

    def send(
        self,
        location,
        garbage_collect_data: bool = True,
    ):
        ptr = self.owner.send(
            self,
            location,
            garbage_collect_data=garbage_collect_data,
        )

        ptr.description = self.description
        ptr.tags = self.tags

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

    @staticmethod
    def detail(worker: AbstractWorker, model: tuple) -> "OnnxModel":
        """
        This function reconstructs an OnnxModel given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            model: a tuple holding the attributes of the OnnxModel
        Returns:
            OnnxModel: an OnnxModel
        """

        (serialized_model, id, tags, description) = model

        model = OnnxModel(
            serialized_model=syft.serde.msgpack.serde._detail(worker, serialized_model),
            id=syft.serde.msgpack.serde._detail(worker, id),
            owner=worker,
            tags=syft.serde.msgpack.serde._detail(worker, tags),
            description=syft.serde.msgpack.serde._detail(worker, description),
        )

        return model

    @staticmethod
    def bufferize(worker, onnx_model):
        """
        This method serializes OnnxModel into OnnxModelPB.
         Args:
            onnx_model (OnnxModel): input OnnxModel to be serialized.
         Returns:
            proto_prec_tensor (FixedPrecisionTensorPB): serialized FixedPrecisionTensor
        """
        proto_onnx_model = OnnxModelPB()
        syft.serde.protobuf.proto.set_protobuf_id(proto_onnx_model.id, onnx_model.id)
        proto_onnx_model.serialized_model = onnx_model.serialized_model
        for tag in onnx_model.tags:
            proto_onnx_model.tags.append(tag)
        proto_onnx_model.description = onnx_model.description

        return proto_onnx_model

    @staticmethod
    def unbufferize(worker, proto_onnx_model):
        """
        This method deserializes OnnxModelPB into OnnxModel.
        Args:
            proto_onnx_model (OnnxModelPB): input OnnxModel to be
            deserialized.
        Returns:
            onnx_model (OnnxModel): deserialized OnnxModelPB
        """
        proto_id = syft.serde.protobuf.proto.get_protobuf_id(proto_onnx_model.id)

        onnx_model = OnnxModel(
            proto_onnx_model.serialized_model,
            owner=worker,
            id=proto_id,
            tags=set(proto_onnx_model.tags),
            description=proto_onnx_model.description,
        )

        return onnx_model

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for OnnxModel.
        Returns:
            Protobuf schema for OnnxModel.
        """
        return OnnxModelPB
