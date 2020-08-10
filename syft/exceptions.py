"""Specific Pysyft exceptions."""
from tblib import Traceback
import traceback
from six import reraise
from typing import Tuple

import syft as sy
from syft.generic.frameworks.types import FrameworkTensor


class DependencyError(Exception):
    def __init__(self, package, pypi_alias=None):
        if pypi_alias is None:
            pypi_alias = package
        message = (
            f"The {package} dependency is not installed. If you intend"
            " to use it, please install it at your command line with "
            f"`pip install {pypi_alias}`."
        )
        super().__init__(message)


class PureFrameworkTensorFoundError(BaseException):
    """Exception raised for errors in the input.
    This error is used in a recursive analysis of the args provided as an
    input of a function, to break the recursion if a FrameworkTensor is found
    as it means that _probably_ all the tensors are pure torch/tensorflow and
    the function can be applied natively on this input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    pass


class RemoteObjectFoundError(BaseException):
    """Exception raised for errors in the input.
    This error is used in a context similar to PureFrameworkTensorFoundError but
    to indicate that a Pointer to a remote tensor was found  in the input
    and thus that the command should be send elsewhere. The pointer retrieved
    by the error gives the location where the command should be sent.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, pointer):
        self.pointer = pointer


class InvalidTensorForRemoteGet(Exception):
    """Raised when a chain of pointer tensors is not provided for `remote_get`."""

    def __init__(self, tensor: object):
        message = (
            "Tensor does not have attribute child. You remote get should "
            f"be called on a chain of pointer tensors, instead you called it on {tensor}."
        )
        super().__init__(message)


class WorkerNotFoundException(Exception):
    """Raised when a non-existent worker is requested."""

    pass


class CompressionNotFoundException(Exception):
    """Raised when a non existent compression/decompression scheme is requested."""

    pass


class CannotRequestObjectAttribute(Exception):
    """Raised when .get() is called on a pointer which points to an attribute of
    another object."""

    pass


class TensorsNotCollocatedException(Exception):
    """Raised when a command is executed on two tensors which are not
    on the same machine. The goal is to provide as useful input as possible
    to help the user identify which tensors are where so that they can debug
    which one needs to be moved."""

    def __init__(self, tensor_a, tensor_b, attr="a method"):

        if hasattr(tensor_a, "child") and tensor_a.is_wrapper:
            tensor_a = tensor_a.child

        if hasattr(tensor_b, "child") and tensor_b.is_wrapper:
            tensor_b = tensor_b.child

        if isinstance(tensor_a, sy.PointerTensor) and isinstance(tensor_b, sy.PointerTensor):
            message = (
                f"You tried to call {attr} involving two tensors which are not on the same machine!"
                " One tensor is on {tensor_a.location} while the other is on {tensor_b.location}."
                " Use a combination of .move(), .get(), and/or .send()"
                " to co-locate them to the same machine."
            )
        elif isinstance(tensor_a, sy.PointerTensor):
            message = (
                f"You tried to call {attr} involving two tensors where one tensor is actually"
                " located on another machine (is a PointerTensor). Call .get() on a"
                " the PointerTensor or .send({tensor_b.location.id}) on the other tensor.\n"
                "Tensor A: {tensor_a}\n"
                "Tensor B: {tensor_b}"
            )
        elif isinstance(tensor_b, sy.PointerTensor):
            message = (
                f"You tried to call {attr} involving two tensors where one tensor is actually"
                " located on another machine (is a PointerTensor). Call .get() on a"
                " the PointerTensor or .send({tensor_a.location.id}) on the other tensor.\n"
                "Tensor A: {tensor_a}\n"
                "Tensor B: {tensor_b}"
            )
        else:
            message = (
                f"You tried to call {attr} involving two tensors which are not"
                " on the same machine. Try calling .send(), .move(), and/or .get() on these"
                " tensors to get them to the same worker before calling methods that involve"
                " them working together."
            )

        super().__init__(message)

        self.tensor_a = tensor_a
        self.tensor_b = tensor_b


class ResponseSignatureError(Exception):
    """Raised when the return of a hooked function is not correctly predicted
    (when defining in advance ids for results)
    """

    def __init__(self, ids_generated=None):
        self.ids_generated = ids_generated

    def get_attributes(self):
        """
        Specify all the attributes need to report an error correctly.
        """
        return {"ids_generated": self.ids_generated}

    @staticmethod
    def simplify(worker: "sy.workers.AbstractWorker", e):
        """
        Serialize information about an Exception which was raised to forward it
        """
        # Get information about the exception: type of error,  traceback
        tp = type(e)
        tb = e.__traceback__
        # Serialize the traceback
        traceback_str = "Traceback (most recent call last):\n" + "".join(traceback.format_tb(tb))
        # Include special attributes if relevant
        try:
            attributes = e.get_attributes()
        except AttributeError:
            attributes = {}
        return (
            sy.serde.msgpack.serde._simplify(worker, tp.__name__),
            sy.serde.msgpack.serde._simplify(worker, traceback_str),
            sy.serde.msgpack.serde._simplify(worker, attributes),
        )

    @staticmethod
    def detail(worker: "sy.workers.AbstractWorker", error_tuple: Tuple[str, str, dict]):
        """
        Detail and re-raise an Exception forwarded by another worker
        """
        error_name, traceback_str, attributes = error_tuple
        error_name = sy.serde.msgpack.serde._detail(worker, error_name)
        traceback_str = sy.serde.msgpack.serde._detail(worker, traceback_str)
        attributes = sy.serde.msgpack.serde._detail(worker, attributes)
        # De-serialize the traceback
        tb = Traceback.from_string(traceback_str)
        # Check that the error belongs to a valid set of Exceptions
        if error_name in dir(sy.exceptions):
            error_type = getattr(sy.exceptions, error_name)
            error = error_type()
            # Include special attributes if any
            for attr_name, attr in attributes.items():
                setattr(error, attr_name, attr)
            reraise(error_type, error, tb.as_traceback())
        else:
            raise ValueError(f"Invalid Exception returned:\n{traceback_str}")


class SendNotPermittedError(Exception):
    """Raised when calling send on a tensor which does not allow
    send to be called on it. This can happen do to sensitivity being too high"""

    @staticmethod
    def simplify(worker: "sy.workers.AbstractWorker", e):
        """
        Serialize information about an Exception which was raised to forward it
        """
        # Get information about the exception: type of error,  traceback
        tp = type(e)
        tb = e.__traceback__
        # Serialize the traceback
        traceback_str = "Traceback (most recent call last):\n" + "".join(traceback.format_tb(tb))
        # Include special attributes if relevant
        try:
            attributes = e.get_attributes()
        except AttributeError:
            attributes = {}
        return tp.__name__, traceback_str, sy.serde.msgpack.serde._simplify(worker, attributes)

    @staticmethod
    def detail(worker: "sy.workers.AbstractWorker", error_tuple: Tuple[str, str, dict]):
        """
        Detail and re-raise an Exception forwarded by another worker
        """
        error_name, traceback_str, attributes = error_tuple
        error_name, traceback_str = error_name.decode("utf-8"), traceback_str.decode("utf-8")
        attributes = sy.serde.msgpack.serde._detail(worker, attributes)
        # De-serialize the traceback
        tb = Traceback.from_string(traceback_str)
        # Check that the error belongs to a valid set of Exceptions
        if error_name in dir(sy.exceptions):
            error_type = getattr(sy.exceptions, error_name)
            error = error_type()
            # Include special attributes if any
            for attr_name, attr in attributes.items():
                setattr(error, attr_name, attr)
            reraise(error_type, error, tb.as_traceback())
        else:
            raise ValueError(f"Invalid Exception returned:\n{traceback_str}")


class GetNotPermittedError(Exception):
    """Raised when calling get on a pointer to a tensor which does not allow
    get to be called on it. This can happen do to sensitivity being too high"""

    @staticmethod
    def simplify(worker: "sy.workers.AbstractWorker", e):
        """
        Serialize information about an Exception which was raised to forward it
        """
        # Get information about the exception: type of error,  traceback
        tp = type(e)
        tb = e.__traceback__
        # Serialize the traceback
        traceback_str = "Traceback (most recent call last):\n" + "".join(traceback.format_tb(tb))
        # Include special attributes if relevant
        try:
            attributes = e.get_attributes()
        except AttributeError:
            attributes = {}
        return (
            sy.serde.msgpack.serde._simplify(worker, tp.__name__),
            sy.serde.msgpack.serde._simplify(worker, traceback_str),
            sy.serde.msgpack.serde._simplify(worker, attributes),
        )

    @staticmethod
    def detail(worker: "sy.workers.AbstractWorker", error_tuple: Tuple[str, str, dict]):
        """
        Detail and re-raise an Exception forwarded by another worker
        """
        error_name, traceback_str, attributes = error_tuple
        error_name = sy.serde.msgpack.serde._detail(worker, error_name)
        traceback_str = sy.serde.msgpack.serde._detail(worker, traceback_str)
        attributes = sy.serde.msgpack.serde._detail(worker, attributes)
        # De-serialize the traceback
        tb = Traceback.from_string(traceback_str)
        # Check that the error belongs to a valid set of Exceptions
        if error_name in dir(sy.exceptions):
            error_type = getattr(sy.exceptions, error_name)
            error = error_type()
            # Include special attributes if any
            for attr_name, attr in attributes.items():
                setattr(error, attr_name, attr)
            reraise(error_type, error, tb.as_traceback())
        else:
            raise ValueError(f"Invalid Exception returned:\n{traceback_str}")


class IdNotUniqueError(Exception):
    """Raised by the ID Provider when setting ids that have already been generated"""

    pass


class PlanCommandUnknownError(Exception):
    """Raised when an unknown plan command execution is requested."""

    def __init__(self, command_name: object):
        message = f"Command {command_name} is not implemented."
        super().__init__(message)


class ObjectNotFoundError(Exception):
    """Raised when object with given object id is not found on worker

    Attributes:
        obj_id -- id of the object with which the interaction is attempted
        worker -- virtual worker on which the interaction is attempted
    """

    def __init__(self, obj_id, worker):
        message = ""
        message += 'Object "' + str(obj_id) + '" not found on worker! '
        message += (
            f"You just tried to interact with an object ID: {obj_id} on {worker} "
            "which does not exist!!!"
        )
        message += (
            "Use .send() and .get() on all your tensors to make sure they're "
            "on the same machines. If you think this tensor does exist, check "
            "the object_store._objects dict on the worker and see for yourself. "
            "The most common reason this error happens is because someone calls "
            ".get() on the object's pointer without realizing it (which deletes "
            "the remote object and sends it to the pointer). Check your code to "
            "make sure you haven't already called .get() on this pointer!"
        )
        super().__init__(message)


class InvalidProtocolFileError(Exception):
    """Raised when PySyft protocol file cannot be loaded."""

    pass


class UndefinedProtocolTypeError(Exception):
    """Raised when trying to serialize type that is not defined in protocol file."""

    pass


class UndefinedProtocolTypePropertyError(Exception):
    """Raised when trying to get protocol type property that is not defined in protocol file."""

    pass


class EmptyCryptoPrimitiveStoreError(Exception):
    """Raised when trying to get crypto primtives from an empty crypto store"""

    def __init__(self, crypto_store=None, available_instances=None, **kwargs):
        if crypto_store is not None:
            message = (
                f"You tried to run a crypto protocol on worker {crypto_store._owner.id} "
                f"but its crypto_store doesn't have enough primitives left for the type "
                f"'{kwargs.get('op')} {kwargs.get('shapes')}' ({kwargs.get('n_instances')}"
                f" were requested while only {available_instances} are available). Use "
                f"your crypto_provider to `provide_primitives` to your worker."
            )
            super().__init__(message)
        self.kwargs_ = kwargs

    @staticmethod
    def simplify(worker: "sy.workers.AbstractWorker", e):
        """
        Serialize information about an Exception which was raised to forward it
        """
        # Get information about the exception: type of error,  traceback
        tp = type(e)
        tb = e.__traceback__
        # Serialize the traceback
        traceback_str = "Traceback (most recent call last):\n" + "".join(traceback.format_tb(tb))
        # Include special attributes if relevant
        attributes = {"kwargs_": e.kwargs_}

        return (
            sy.serde.msgpack.serde._simplify(worker, tp.__name__),
            sy.serde.msgpack.serde._simplify(worker, traceback_str),
            sy.serde.msgpack.serde._simplify(worker, attributes),
        )

    @staticmethod
    def detail(worker: "sy.workers.AbstractWorker", error_tuple: Tuple[str, str, dict]):
        """
        Detail and re-raise an Exception forwarded by another worker
        """
        error_name, traceback_str, attributes = error_tuple
        error_name = sy.serde.msgpack.serde._detail(worker, error_name)
        traceback_str = sy.serde.msgpack.serde._detail(worker, traceback_str)
        attributes = sy.serde.msgpack.serde._detail(worker, attributes)
        # De-serialize the traceback
        tb = Traceback.from_string(traceback_str)
        # Check that the error belongs to a valid set of Exceptions
        if error_name in dir(sy.exceptions):
            error_type = getattr(sy.exceptions, error_name)
            error = error_type()
            # Include special attributes if any
            for attr_name, attr in attributes.items():
                setattr(error, attr_name, attr)
            reraise(error_type, error, tb.as_traceback())
        else:
            raise ValueError(f"Invalid Exception returned:\n{traceback_str}")


class TranslationUnavailableError(Exception):
    """Raised when trying to translate a plan to use a framework that is unavailable"""

    pass


def route_method_exception(exception, self, args_, kwargs_):
    try:
        if self.is_wrapper and isinstance(self.child, sy.PointerTensor) and len(args_) > 0:
            if not args_[0].is_wrapper:
                return TensorsNotCollocatedException(self, args_[0])
            elif isinstance(args_[0].child, sy.PointerTensor):
                if self.location != args_[0].child.location:
                    return TensorsNotCollocatedException(self, args_[0])

        # if self is a normal tensor
        elif isinstance(self, FrameworkTensor) and len(args_) > 0:
            if args_[0].is_wrapper and isinstance(args_[0].child, sy.PointerTensor):
                return TensorsNotCollocatedException(self, args_[0])
            elif isinstance(args_[0], sy.PointerTensor):
                return TensorsNotCollocatedException(self, args_[0])
    except:  # noqa: E722
        ""
    return exception
