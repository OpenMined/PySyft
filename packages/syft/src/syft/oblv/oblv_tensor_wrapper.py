# stdlib
import sys
from typing import Any
from typing import Union

# third party
import numpy as np
import torch

# relative
from ..core.common.serde.serialize import _serialize as serialize
from ..core.tensor import Tensor
from ..util import bcolors


class OblvTensorWrapper:
    def __init__(self, id, deployment_client):
        self.id = id
        self.deployment_client = deployment_client

    def request_publish(self, sigma=0.5):
        return self.deployment_client.request_publish(self.id, sigma)

    def _apply_op(
        self,
        other: Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"],
        op_str: str,
    ):
        """
        Apply a given operation.

        Args:
            op_str (str): Operator.
            *args (Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]): tensor to apply the operator.

        Returns:
            OblvTensorWrapper: Result of the operation.
        """
        arguments = [{"type": "wrapper", "value": self.id}]
        # Adding the other argument
        type_name = type(other)
        if type_name == int:
            arg = {"type": "int", "value": other}
        elif type_name == float:
            arg = {"type": "float", "value": other}
        elif type_name == torch.Tensor or type_name == np.ndarray:
            t = Tensor(other)
            arg = {"type": "tensor", "value": serialize(t, to_bytes=True)}
        elif type_name == OblvTensorWrapper:
            arg = {"type": "wrapper", "value": other.id}
        else:
            print(
                bcolors.RED
                + bcolors.BOLD
                + "Exception"
                + bcolors.BLACK
                + bcolors.ENDC
                + ": "
                + "Argument of invalid type",
                file=sys.stderr,
            )
            return
        arguments.append(arg)
        new_id = self.deployment_client.publish_action(op_str, arguments)
        result = OblvTensorWrapper(id=new_id, deployment_client=self.deployment_client)
        return result

    def _apply_self_tensor_op(self, op_str: str, *args: Any, **kwargs: Any):
        arguments = [{"type": "wrapper", "value": self.id}]
        new_id = self.deployment_client.publish_action(
            op_str, arguments, *args, **kwargs
        )
        result = OblvTensorWrapper(id=new_id, deployment_client=self.deployment_client)
        return result

    def __add__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__add__")

    def sum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Sum of array elements over a given axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a sum is performed.
                The default, axis=None, will sum all of the elements of the input array.
                If axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a
                single axis or all the axes as before.
            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the sum method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.
            initial: scalar, optional
                Starting value for the sum. See reduce for details.
            where: array_like of bool, optional
                Elements to include in the sum. See reduce for details.
        """
        return self._apply_self_tensor_op("sum", *args, **kwargs)
