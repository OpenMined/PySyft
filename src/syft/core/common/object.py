# external class/method imports (sorted by length)
from ...proto.core.common.common_object_pb2 import ObjectWithID as ObjectWithID_PB

# syft imports (sorted by length)
from ...decorators.syft_decorator_impl import syft_decorator
from .serializable import Serializable
from .uid import UID


class AbstractObjectWithID(Serializable):
    """This exists to allow us to typecheck on the ObjectWithId object
    because we need a type which has already been initialized in
    order to add it as a type hint on the ObjectWithId object.
    """


class ObjectWithID(AbstractObjectWithID):
    """This object is the superclass for nearly all Syft objects. Subclassing
    from this object will cause an object to be initialized with a unique id
    using the process specified in the UID class.

    .. note::
        At the time of writing, the only class in Syft which doesn't have an ID
        of some kind is the Client class because it's job is to point to another
        object (which has an ID).

    """

    @syft_decorator(typechecking=True)
    def __init__(self, id: UID = None, as_wrapper: bool = False):
        """This initializer only exists to set the id attribute, which is the
        primary purpose of this class. It also sets the 'as_wrapper' flag
        for the 'Serializable' superclass.

        :param id: an override which can be used to set an ID for this object
            manually. This is probably only used for deserialization.
        :type id: UID
        :param as_wrapper: this flag determines whether the subclass can also
            be used as a wrapper class around a non-syft object. For details on
            why, see :py:mod:`syft.core.common.serializable.Serializable`.

        """

        # while this class is never used as a simple wrapper,
        # it's possible that sub-classes of this class will be.
        super().__init__(as_wrapper=as_wrapper)

        if id is None:
            id = UID()

        self.id = id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> ObjectWithID_PB:
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
        self_type = type(self)
        obj_type = self_type.__module__ + "." + self_type.__name__
        return ObjectWithID_PB(
            obj_type=obj_type, id=self.id.serialize(), as_wrapper=self.as_wrapper
        )

    @staticmethod
    def _proto2object(proto: ObjectWithID_PB) -> AbstractObjectWithID:
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ObjectWithID
        :rtype: ObjectWithID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return ObjectWithID(id=UID.deserialize(proto.id))
