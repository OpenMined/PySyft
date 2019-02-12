"""Specific Pysyft exceptions."""

import syft as sy
import torch


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

    def __init__(self, tensor):
        self.tensor = tensor


class RemoteTensorFoundError(BaseException):
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


class WorkerNotFoundException(Exception):
    """Raised when a non-existent worker is requested."""

    pass


class CompressionNotFoundException(Exception):
    """Raised when a non existent compression/decompression scheme is requested."""

    pass


class CannotRequestTensorAttribute(Exception):
    """Raised when .get() is called on a pointer which points to an attribute of
    another tensor."""

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
                + " involving two tensors which "
                + "are not on the same machine! One tensor is on "
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
                + "on another machine (is a PointerTensor). Call .get() on the PointerTensor or .send("
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
                + "on another machine (is a PointerTensor). Call .get() on the PointerTensor or .send("
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
