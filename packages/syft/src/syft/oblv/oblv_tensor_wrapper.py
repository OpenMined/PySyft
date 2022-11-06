# stdlib
import base64
from copy import deepcopy
import functools
from functools import lru_cache
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union
from uuid import uuid4

# third party
import numpy as np
import torch

# relative
from ..core.tensor import Tensor


class OblvTensorWrapper():
    

    def __init__(self, id,deployment_client):
        self.id = id
        self.deployment_client=deployment_client

    def apply_function(
        self,
        op_str: str,
        *args
    ):
        """
        Apply a given operation.

        Args:
            op_str (str): Operator.
            *args (Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]): tensor to apply the operator.

        Returns:
            OblvTensorWrapper: Result of the operation.
        """
        arguments = [{
            "type": "wrapper",
            "value": self.id
        }]
        for a in args:
            type_name = type(a)
            if type_name==int:
                arg = {
                    "type": "int",
                    "value": a
                }
            elif type_name==float:
                arg = {
                    "type": "float",
                    "value": a
                }
            elif type_name==torch.Tensor or type_name==np.ndarray:
                t = Tensor(a)
                arg = {
                    "type": "tensor",
                    "value": base64.b64encode(t._object2bytes()).decode('ASCII')
                }
            elif type_name==OblvTensorWrapper:
                arg = {
                    "type": "wrapper",
                    "value": a.id
                }
            arguments.append(arg)
        new_id = self.deployment_client.publish_action(op_str,arguments)
        result = OblvTensorWrapper(id=new_id, deployment_client=self.deployment_client)
        return result

    def request_publish(self,sigma = 0.5):
        return self.deployment_client.request_publish(self.id,sigma)

    def add(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]
    ):
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): self + y

        Returns:
            ShareTensor. Result of the operation.
        """
        new_share = self.apply_function("__add__", y)
        return new_share
    
    def sub(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]
    ):
        """Apply the "sub" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]): self - y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        new_share = self.apply_function("__sub__", y)
        return new_share

    def rsub(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]
    ):
        """Apply the "rsub" operation between "self" and "y"

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]): y - self

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        new_share = self.apply_function("__rsub__", y)
        return new_share

    def mul(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]
    ):
        """Apply the "mul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]): self * y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        new_share = self.apply_function("__mul__", y)
        return new_share

    def truediv(self, y: Union[int, float, np.ndarray, "OblvTensorWrapper"]):
        """Apply the "division" operation between "self" and "y".

        Args:
            y (Union[int, float, np.ndarray, "OblvTensorWrapper"]): self / y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """

        if not isinstance(y, (int, np.integer)):
            raise ValueError("Current Division only works for integers")
        else:
            new_share = self.apply_function("__truediv__" , y)

        return new_share

    def matmul(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]
    ):
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "OblvTensorWrapper"]): self @ y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        new_share = self.apply_function("__matmul__", y)

        return new_share

    def rmatmul(self, y: torch.Tensor):
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y (torch.Tensor): y @ self

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        if isinstance(y, OblvTensorWrapper):
            raise ValueError("Private matmul not supported yet")

        new_share = self.apply_function("__rmatmul__",y)
        return new_share

    def lt(self, y: Union["OblvTensorWrapper", np.ndarray]):
        """Apply the "lt" operation between "y" and "self".

        Args:
            y (Union[OblvTensorWrapper,np.ndarray]): self < y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = self.apply_function("__lt__",y)
        return new_share

    def gt(self, y: Union["OblvTensorWrapper", np.ndarray]):
        """Apply the "gt" operation between "y" and "self".

        Args:
            y (Union[OblvTensorWrapper,np.ndarray]): self > y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = self.apply_function("__gt__",y)
        return new_share

    def ge(self, y: Union["OblvTensorWrapper", np.ndarray]):
        """Apply the "ge" operation between "y" and "self".

        Args:
            y (Union[OblvTensorWrapper,np.ndarray]): self >= y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = self.apply_function("__le__",y)
        return new_share

    def le(self, y: Union["OblvTensorWrapper", np.ndarray]):
        """Apply the "le" operation between "y" and "self".

        Args:
            y (Union[OblvTensorWrapper,np.ndarray]): self <= y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = self.apply_function("__le__",y)
        return new_share

    def ne(self, y: Union["OblvTensorWrapper", np.ndarray]):
        """Apply the "ne" operation between "y" and "self".

        Args:
            y (Union[OblvTensorWrapper,np.ndarray]): self != y

        Returns:
            OblvTensorWrapper. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = self.apply_function("__ne__",y)
        return new_share

    def eq(self, other: Any):
        """Equal operator.
        Check if "self" is equal with another object given a set of
            attributes to compare.
        Args:
            other (Any): Value to compare.
        Returns:
            bool: True if equal False if not.
        """
        res = self.apply_function("__eq__",other)
        return res

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = truediv
    __lt__ = lt
    __gt__ = gt
    __ge__ = ge
    __le__ = le
    __eq__ = eq
    __ne__ = ne
