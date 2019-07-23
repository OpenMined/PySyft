"""Specific Pysyft exceptions."""

import syft as sy
import torch
from tblib import Traceback
import traceback
from six import reraise
from typing import Tuple


class PureTorchTensorFoundError(BaseException):
    """Exception raised for errors in the input.
    This error is used in a recursive analysis of the args provided as an
    input of a function, to break the recursion if a TorchTensor is found
    as it means that _probably_ all the tensors are pure torch tensor and
    the function can be applied natively on this input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    pass


class RemoteObjectFoundError(BaseException):
    """Exception raised for errors in the input.
    This error is used in a context similar to PureTorchTensorFoundError but
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
        message = "Tensor does not have attribute child. You remote get should be called on a chain of pointer tensors, instead you called it on {}.".format(
            tensor
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
                "You tried to call "
                + attr
                + " involving two tensors which"
                + " are not on the same machine! One tensor is on "
                + str(tensor_a.location)
                + " while the other is on "
                + str(tensor_b.location)
                + ". Use a combination of .move(), .get(), and/or .send() to co-locate them to the same machine."
            )
        elif isinstance(tensor_a, sy.PointerTensor):
            message = (
                "You tried to call "
                + attr
                + " involving two tensors where one tensor is actually located"
                + " on another machine (is a PointerTensor). Call .get() on the PointerTensor or .send("
                + str(tensor_a.location.id)
                + ") on the other tensor.\n"
                + "\nTensor A: "
                + str(tensor_a)
                + "\nTensor B: "
                + str(tensor_b)
            )
        elif isinstance(tensor_b, sy.PointerTensor):
            message = (
                "You tried to call "
                + attr
                + " involving two tensors where one tensor is actually located"
                + " on another machine (is a PointerTensor). Call .get() on the PointerTensor or .send("
                + str(tensor_b.location.id)
                + ") on the other tensor.\n"
                + "\nTensor A: "
                + str(tensor_a)
                + "\nTensor B: "
                + str(tensor_b)
            )
        else:
            message = (
                "You tried to call "
                + attr
                + " involving two tensors which are not on the same machine."
                + "Try calling .send(), .move(), and/or .get() on these tensors to get them to the same"
                + "worker before calling methods that involve them working together."
            )

        super().__init__(message)

        self.tensor_a = tensor_a
        self.tensor_b = tensor_b


class ResponseSignatureError(Exception):
    """Raised when the return of a hooked function is not correctly predicted
    (when defining in advance ids for results)"""

    def __init__(self, ids_generated=None):
        self.ids_generated = ids_generated

    def get_attributes(self):
        """
        Specify all the attributes need to report an error correctly.
        """
        return {"ids_generated": self.ids_generated}

    @staticmethod
    def simplify(e):
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
        return tp.__name__, traceback_str, sy.serde._simplify(attributes)

    @staticmethod
    def detail(worker: "sy.workers.AbstractWorker", error_tuple: Tuple[str, str, dict]):
        """
        Detail and re-raise an Exception forwarded by another worker
        """
        error_name, traceback_str, attributes = error_tuple
        error_name, traceback_str = error_name.decode("utf-8"), traceback_str.decode("utf-8")
        attributes = sy.serde._detail(worker, attributes)
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
    def simplify(e):
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
        return tp.__name__, traceback_str, sy.serde._simplify(attributes)

    @staticmethod
    def detail(worker: "sy.workers.AbstractWorker", error_tuple: Tuple[str, str, dict]):
        """
        Detail and re-raise an Exception forwarded by another worker
        """
        error_name, traceback_str, attributes = error_tuple
        error_name, traceback_str = error_name.decode("utf-8"), traceback_str.decode("utf-8")
        attributes = sy.serde._detail(worker, attributes)
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


def route_method_exception(exception, self, args, kwargs):
    try:
        if self.is_wrapper:
            if isinstance(self.child, sy.PointerTensor):
                if len(args) > 0:
                    if not args[0].is_wrapper:
                        return TensorsNotCollocatedException(self, args[0])
                    elif isinstance(args[0].child, sy.PointerTensor):
                        if self.location != args[0].child.location:
                            return TensorsNotCollocatedException(self, args[0])

        # if self is a normal tensor
        elif isinstance(self, torch.Tensor):
            if len(args) > 0:
                if args[0].is_wrapper:
                    if isinstance(args[0].child, sy.PointerTensor):
                        return TensorsNotCollocatedException(self, args[0])
                elif isinstance(args[0], sy.PointerTensor):
                    return TensorsNotCollocatedException(self, args[0])
    except:
        ""
    return exception
