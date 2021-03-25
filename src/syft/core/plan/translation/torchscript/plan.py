# stdlib
import io

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syft_proto.execution.v1.plan_pb2 import Plan as Plan_PB
import torch as th

# syft absolute
from syft.core.common.object import Serializable
from syft.logger import traceback_and_raise


class PlanTorchscript(Serializable):
    """
    Represents Syft Plan translated to TorchScript
    """

    def __init__(self, torchscript=None):
        super().__init__()
        self.torchscript = torchscript

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """
        return Plan_PB

    def __call__(self, *args):
        if self.torchscript:
            return self.torchscript(*args)
        else:
            traceback_and_raise("No torchscript")

    def _object2proto(self) -> Plan_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectWithID_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        bin = self.torchscript.save_to_buffer() if self.torchscript else None
        return Plan_PB(torchscript=bin)

    @staticmethod
    def _proto2object(proto: Plan_PB) -> "PlanTorchscript":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Plan
        :rtype: Plan

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        if proto.torchscript:
            bin = io.BytesIO(proto.torchscript)
            ts = th.jit.load(bin)
        else:
            ts = None

        return PlanTorchscript(torchscript=ts)
