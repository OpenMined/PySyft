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

    def sum(
        self
    ) -> "OblvTensorWrapper":
        """
        Sum of array elements
        """
        return self.apply_function("sum")

    def __lshift__(
        self,
        other: Union[
            "OblvTensorWrapper", int, float, np.ndarray
        ],
    ) -> "OblvTensorWrapper":
        """Apply the "lshift" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            "OblvTensorWrapper" : Result of the operation.
        """
        return self.apply_function(other, "__lshift__")

    def __rshift__(
        self,
        other: Union[
            "OblvTensorWrapper", int, float, np.ndarray
        ],
    ) -> "OblvTensorWrapper":
        """Apply the "rshift" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            "OblvTensorWrapper" : Result of the operation.
        """
        return self.apply_function(other, "__rshift__")

    def __abs__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "abs" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            "OblvTensorWrapper" : Result of the operation.
        """
        return self.apply_function("__abs__")


    def __pos__(self) -> "OblvTensorWrapper":
        """Apply the pos (+) operator  on self.

        Returns:
            "OblvTensorWrapper" : Result of the operation.
        """
        return self.apply_function("__pos__")

    def cumsum(
        self
    ) -> "OblvTensorWrapper":
        """ "
        Return the cumulative sum of the elements 
        """
        return self.apply_function("cumsum")

    def cumprod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return the cumulative product of the elements
        """
        return self.apply_function("cumprod")

    def prod(
        self
    ) -> "OblvTensorWrapper":
        """
        Return the product of array elements
        """
        return self.apply_function("prod")

    def __xor__(
        self,
        other: Union[
            "OblvTensorWrapper", int, float, np.ndarray
        ],
    ) -> "OblvTensorWrapper":
        """Apply the "xor" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            "OblvTensorWrapper" : Result of the operation.
        """
        return self.apply_function("__xor__", other)

    def mean(self) -> "OblvTensorWrapper":
        """
        Compute the arithmetic mean 
        """
        return self.apply_function("mean")

    def std(
        self
    ) -> "OblvTensorWrapper":
        """
        Compute the standard deviation 
        """
        return self.apply_function("std")

    def min(
        self
    ) -> "OblvTensorWrapper":
        """
        Return the minimum of an array 
        """
        return self.apply_function("min")

    def max(
        self
    ) -> "OblvTensorWrapper":
        """
        Return the maximum of an array
        """
        return self.apply_function("max")

    def flatten(
        self
    ) -> "OblvTensorWrapper":
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
            order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
            ‘C’ means to flatten in row-major (C-style) order.
            ‘F’ means to flatten in column-major (Fortran- style) order.
            ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
            ‘K’ means to flatten a in the order the elements occur in memory. The default is ‘C’.

        Returns
            y: PhiTensor
                A copy of the input array, flattened to one dimension.
        """
        return self.apply_function("flatten")

    

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
