# stdlib
import sys
from typing import Any
from typing import Tuple
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

    def __sub__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__sub__")

    def __mul__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__mul__")

    def __matmul__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "matmul" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__matmul__")

    def __rmatmul__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "rmatmul" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__rmatmul__")

    def __lt__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__lt__")

    def __gt__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__gt__")

    def __ge__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__ge__")

    def __le__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__le__")

    def __eq__(  # type: ignore
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__eq__")

    def __ne__(  # type: ignore
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__ne__")

    # Concatenate only available for MPC tensor, need to implement for Gamma Tensor
    # def concatenate(
    #     self,
    #     other: "OblvTensorWrapper",
    #     *args: Any,
    #     **kwargs: Any,
    # ) -> "OblvTensorWrapper":
    #     """Apply the "concatenate" operation between "self" and "other"

    #     Args:
    #         y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.
    #     Returns:
    #         Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
    #     """
    #     if not isinstance(other, OblvTensorWrapper):
    #         raise ValueError(
    #             f"Concatenate works only for OblvTensorWrapper got type: {type(other)}"
    #         )
    #     if self.client != other.client:
    #         parties = [self.client, other.client]
    #         self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
    #         other_mpc = MPCTensor(
    #             secret=other, shape=other.public_shape, parties=parties
    #         )
    #         return self_mpc.concatenate(other_mpc, *args, **kwargs)
    #     else:
    #         raise ValueError(
    #             "Concatenate method currently works only between two different clients."
    #         )

    def __truediv__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[OblvTensorWrapper,int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__truediv__")

    def __mod__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        return OblvTensorWrapper._apply_op(self, other, "__mod__")

    def __and__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "and" operation between "self" and "other"

        Args:
            y (Union[OblvTensorWrapper,int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__and__")

    def __or__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "or" operation between "self" and "other"

        Args:
            y (Union[OblvTensorWrapper,int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__or__")

    def __floordiv__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "floordiv" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__floordiv__")

    def __divmod__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> Tuple["OblvTensorWrapper", "OblvTensorWrapper",]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self.divmod(other)

    def divmod(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> Tuple["OblvTensorWrapper", "OblvTensorWrapper",]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(
            self, other, "__floordiv__"
        ), OblvTensorWrapper._apply_op(self, other, "__mod__")

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

    def ptp(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "ptp" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("ptp", *args, **kwargs)

    def __lshift__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "lshift" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__lshift__")

    def argmax(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "argmax" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("argmax", *args, **kwargs)

    def __rshift__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "rshift" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__rshift__")

    def argmin(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "argmin" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("argmin", *args, **kwargs)

    def __abs__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "abs" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("__abs__", *args, **kwargs)

    def all(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "all" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("all", *args, **kwargs)

    def any(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "any" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("any", *args, **kwargs)

    def round(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        return self._apply_self_tensor_op("round", *args, **kwargs)

    def __round__(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        return self.round(*args, **kwargs)

    def __pos__(self) -> "OblvTensorWrapper":
        """Apply the pos (+) operator  on self.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op(op_str="__pos__")

    def var(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Compute the variance along the specified axis of the array elements, a measure of the spread of a distribution.
        The variance is computed for the flattened array by default, otherwise over the specified axis.

        Parameters

            axis: None or int or tuple of ints, optional
                Axis or axes along which the variance is computed.
                The default is to compute the variance of the flattened array.
                If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all
                the axes as before.

            ddof: int, optional
                “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof, where N represents the
                number of elements. By default ddof is zero.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the var method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the variance. See reduce for details.
        """
        return self._apply_self_tensor_op("var", *args, **kwargs)

    def cumsum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """ "
        Return the cumulative sum of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the
                flattened array.
        Returns
            cumsum_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        return self._apply_self_tensor_op("cumsum", *args, **kwargs)

    def cumprod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return the cumulative product of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over
                the flattened array.
        Returns
            cumprod_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        return self._apply_self_tensor_op("cumprod", *args, **kwargs)

    def prod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return the product of array elements over a given axis.
        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a product is performed.
                The default, axis=None, will calculate the product of all the elements in the input array.
                If axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple instead of
                a single axis or all the axes as before.
            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the prod method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.
            initial: scalar, optional
                The starting value for this product. See reduce for details.
            where: array_like of bool, optional
                Elements to include in the product. See reduce for details.
        """
        return self._apply_self_tensor_op("prod", *args, **kwargs)

    def __xor__(
        self,
        other: Union["OblvTensorWrapper", int, float, np.ndarray],
    ) -> "OblvTensorWrapper":
        """Apply the "xor" operation between "self" and "other"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return OblvTensorWrapper._apply_op(self, other, "__xor__")

    def __pow__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        First array elements raised to powers from second array, element-wise.

        Raise each base in x1 to the positionally-corresponding power in x2.
        x1 and x2 must be broadcastable to the same shape.
        An integer type raised to a negative integer power will raise a ValueError.
        Negative values raised to a non-integral value will return nan.

        Parameters
            x2: array_like

                The exponents. If self.shape != x2.shape, they must be broadcastable to a common shape.

            where: array_like, optional

                This condition is broadcast over the input. At locations where the condition is True, the out array will
                 be set to the ufunc result.
                 Elsewhere, the out array will retain its original value.

            **kwargs
                For other keyword-only arguments, see the ufunc docs.

        Returns
            y: PhiTensorPointer
                The bases in the tensor raised to the exponents in x2. This is a scalar if both self and x2 are scalars.
        """
        return self._apply_self_tensor_op("__pow__", *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array elements. The average is taken over the flattened array by default, otherwise
        over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to compute the mean of the flattened
                array.
        """
        return self._apply_self_tensor_op("mean", *args, **kwargs)

    def std(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Compute the standard deviation along the specified axis.
        Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
        The standard deviation is computed for the flattened array by default, otherwise over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the standard deviation is computed.
                The default is to compute the standard deviation of the flattened array.
                If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single
                axis or all the axes as before.

            out: ndarray, optional
                Alternative output array in which to place the result. It must have the same shape as the expected
                output but the type (of the calculated values) will be cast if necessary.

            ddof: int, optional
                ddof = Delta Degrees of Freedom. By default ddof is zero.
                The divisor used in calculations is N - ddof, where N represents the number of elements.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.

                If the default value is passed, then keepdims will not be passed through to the std method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the standard deviation. See reduce for details.

        Returns

            standard_deviation: PhiTensor
        """
        return self._apply_self_tensor_op("std", *args, **kwargs)

    def trace(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return the sum along diagonals of the array.

        If a is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum of elements
        a[i,i+offset] for all i.

        If a has more than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
        sub-arrays whose traces are returned. The shape of the resulting array is the same as that of a with axis1 and
        axis2 removed.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.

            axis1, axis2: int, optional
                Axes to be used as the first and second axis of the 2-D sub-arrays from which the diagonals should be
                taken. Defaults are the first two axes of a.

        Returns

            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.

        """
        return self._apply_self_tensor_op("trace", *args, **kwargs)

    def sort(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Return a sorted copy of an array.

        Parameters

            a: array_like
                Array to be sorted.

            axis: int or None, optional
                Axis along which to sort. If None, the array is flattened before sorting.
                The default is -1, which sorts along the last axis.

            kind{‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’.
                Note that both ‘stable’ and ‘mergesort’ use timsort or radix sort under the covers and, in general,
                the actual implementation will vary with data type. The ‘mergesort’ option is retained for backwards
                compatibility.

                Changed in version 1.15.0.: The ‘stable’ option was added.

            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare first, second,
                etc. A single field can be specified as a string, and not all fields need be specified, but unspecified
                 fields will still be used, in the order in which they come up in the dtype, to break ties.

        Please see docs here: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        return self._apply_self_tensor_op("sort", *args, **kwargs)

    def argsort(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Returns the indices that would sort an array.

        Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
        It returns an array of indices of the same shape as a that index data along the given axis in sorted order.

        Parameters
            axis: int or None, optional
                Axis along which to sort. The default is -1 (the last axis). If None, the flattened array is used.
            kind: {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’. Note that both ‘stable’ and ‘mergesort’ use timsort
                under the covers and, in general, the actual implementation will vary with data type. The ‘mergesort’
                option is retained for backwards compatibility.
            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare 1st, 2nd, etc.
                A single field can be specified as a string, and not all fields need be specified, but unspecified
                fields will still be used, in the order in which they come up in the dtype, to break ties.

        Returns
            index_array: ndarray, int
                Array of indices that sort a along the specified axis. If a is one-dimensional, a[index_array] yields a
                sorted a. More generally, np.take_along_axis(a, index_array, axis=axis) always yields the sorted a,
                irrespective of dimensionality.
        """
        return self._apply_self_tensor_op("argsort", *args, **kwargs)

    def min(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_min: PhiTensor
                Minimum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        return self._apply_self_tensor_op("min", *args, **kwargs)

    def max(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return the maximum of an array or along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_max: PhiTensor
                Maximum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        return self._apply_self_tensor_op("max", *args, **kwargs)

    def compress(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Return selected slices of an array along given axis.

        When working along a given axis, a slice along that axis is returned in output for each index
        where condition evaluates to True. When working on a 1-D array, compress is equivalent to extract.

        Parameters
            condition: 1-D array of bools
            Array that selects which entries to return. If len(condition) is less than the size of
            a along the given axis,then output is truncated to the length of the condition array.

            axis: int, optional
            Axis along which to take slices. If None (default), work on the flattened array.

        Returns:
            compressed_array: PhiTensor
            A copy of a without the slices along axis for which condition is false.
        """
        return self._apply_self_tensor_op("compress", *args, **kwargs)

    def squeeze(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Remove axes of length one from a.

        Parameters
            axis: None or int or tuple of ints, optional
                Selects a subset of the entries of length one in the shape.
                If an axis is selected with shape entry greater than one, an error is raised.

        Returns:
            squeezed: PhiTensor
                The input array, but with all or a subset of the dimensions of length 1 removed.
                This is always a itself or a view into a.
                Note that if all axes are squeezed, the result is a 0d array and not a scalar.
        """
        return self._apply_self_tensor_op("squeeze", *args, **kwargs)

    def __getitem__(self, key: Union[int, bool, slice]) -> "OblvTensorWrapper":
        """Return self[key].
        Args:
            y (Union[int,bool,slice]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op("__getitem__", key)

    def ones_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the "ones like" operation on self"

        Args:
            y (Union["OblvTensorWrapper",int,float,np.ndarray]) : second operand.

        Returns:
            OblvTensorWrapper : Result of the operation.
        """
        return self._apply_self_tensor_op("ones_like", *args, **kwargs)

    def transpose(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Reverse or permute the axes of an array; returns the modified array.

        Returns
            p: ndarray
                array with its axes permuted. A view is returned whenever possible.
        """

        return self._apply_self_tensor_op("transpose", *args, **kwargs)

    def resize(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":

        """
        Return a new array with the specified shape.

        Parameters
            new_shape: int or tuple of int
                Shape of resized array.

        Returns
            reshaped_array: ndarray
                The new array is formed from the data in the old array,
                repeated if necessary to fill out the required number of elements.
                The data are repeated iterating over the array in C-order.

        """
        return self._apply_self_tensor_op("resize", *args, **kwargs)

    def reshape(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":

        """
        Gives a new shape to an array without changing its data.

        Parameters
            new_shape: int or tuple of int
                The new shape should be compatible with the original shape. If an integer, then the result will
                be a 1-D array of that length. One shape dimension can be -1. In this case,
                the value is inferred from the length of the array and remaining dimensions.

        Returns
            reshaped_array: ndarray
                This will be a new view object if possible; otherwise, it will be a copy.
                Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.
        """
        return self._apply_self_tensor_op("reshape", *args, **kwargs)

    def repeat(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """Apply the repeat" operation

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("repeat", *args, **kwargs)

    def diagonal(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Return specified diagonals.
        If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of elements
        of the form a[i, i+offset].

        If a has more than two dimensions, then the axes specified by axis1 and axis are used to determine
        the 2-D sub-array whose diagonal is returned.  The shape of the resulting array can be determined by
        removing axis1 and axis2 and appending an index to the right equal to the size of the resulting diagonals.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal.  Can be positive or negative.
                Defaults to main diagonal (0).
            axis1, axis2: int, optional
                Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken.
                Defaults are the first two axes of a.

        Returns
            array_of_diagonals : Union[TensorWrappedPhiTensorPointer,MPCTensor]
                If a is 2-D, then a 1-D array containing the diagonal and of the same type as a is returned unless
                a is a matrix, in which case
                a 1-D array rather than a (2-D) matrix is returned in order to maintain backward compatibility.

                If a.ndim > 2, then the dimensions specified by axis1 and axis2 are removed, and a new axis
                inserted at the end corresponding to the diagonal.
        """
        return self._apply_self_tensor_op("diagonal", *args, **kwargs)

    def flatten(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
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
        return self._apply_self_tensor_op("flatten", *args, **kwargs)

    def ravel(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Return a contiguous flattened array.

        A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

        As of NumPy 1.10, the returned array will have the same type as the input array.
        (for example, a masked array will be returned for a masked array input)
        Parameters
            order: {‘C’,’F’, ‘A’, ‘K’}, optional
            The elements of a are read using this index order.
            ‘C’ means to index the elements in row-major,
            C-style order, with the last axis index changing fastest, back to the first axis index changing slowest.
            ‘F’ means to index the elements in column-major, Fortran-style order, with the first index changing fastest,
             and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array,
             and only refer to the order of axis indexing.
            ‘A’ means to read the elements in Fortran-like index order if a is Fortran contiguous in memory,
             C-like order otherwise.
            ‘K’ means to read the elements in the order they occur in memory, except for reversing the data
             when strides are negative.
            By default, ‘C’ index order is used.

        Returns:
            y: PhiTensor
                y is an array of the same subtype as a, with shape (a.size,).
                Note that matrices are special cased for backward compatibility,
                if a is a matrix, then y is a 1-D ndarray.
        """
        return self._apply_self_tensor_op("ravel", *args, **kwargs)

    def take(self, *args: Any, **kwargs: Any) -> "OblvTensorWrapper":
        """
        Take elements from an array along an axis.

        When axis is not None, this function does the same thing as “fancy” indexing (indexing arrays using arrays);
        however, it can be easier to use if you need elements along a given axis.
        A call such as np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].

        Explained without fancy indexing, this is equivalent to the following use of ndindex, \
        which sets each of ii, jj, and kk to a tuple of indices:

            Ni, Nk = a.shape[:axis], a.shape[axis+1:]
            Nj = indices.shape
            for ii in ndindex(Ni):
                for jj in ndindex(Nj):
                    for kk in ndindex(Nk):
                        out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

        Parameters
            indices: array_like (Nj…)
                The indices of the values to extract.

            axis: int, optional
                The axis over which to select values. By default, the flattened input array is used.

            mode: {‘raise’, ‘wrap’, ‘clip’}, optional
                Specifies how out-of-bounds indices will behave.

                * ‘raise’ – raise an error (default)

                * ‘wrap’ – wrap around

                * ‘clip’ – clip to the range

                ‘clip’ mode means that all indices that are too large are replaced by the index
                that addresses the last element along that axis.
                Note that this disables indexing with negative numbers.

        Returns
            out: PhiTensor
                The returned array has the same type as a.
        """
        return self._apply_self_tensor_op("take", *args, **kwargs)

    def clip(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Clip (limit) the values in an array.

        Parameters
            a : array_like
                Array containing elements to clip.
            a_min, a_max : array_like or None
                Minimum and maximum value. If None, clipping is not performed on
                the corresponding edge. Only one of a_min and a_max may be
                None. Both are broadcast against a.
        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("clip", *args, **kwargs)

    def choose(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "OblvTensorWrapper":
        """
        Construct an array from an index array and a list of arrays to choose from.

        First of all, if confused or uncertain, definitely look at the Examples - in its full generality,
        this function is less simple than it might seem from the following code description
        (below ndi = numpy.lib.index_tricks):

        np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

        But this omits some subtleties. Here is a fully general summary:

        Given an “index” array (a) of integers and a sequence of n arrays (choices), a and each choice array are first
        broadcast, as necessary, to arrays of a common shape; calling these Ba and Bchoices[i], i = 0,…,n-1 we have that
         necessarily, Ba.shape == Bchoices[i].shape for each i. Then, a new array with shape Ba.shape is created
         as follows:

            if mode='raise' (the default), then, first of all, each element of a (and thus Ba) must be in the range
            [0, n-1]; now, suppose that i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
            then the value at the same position in the new array is the value in Bchoices[i] at that same position;

            if mode='wrap', values in a (and thus Ba) may be any (signed) integer; modular arithmetic is used to map
            integers outside the range [0, n-1] back into that range; and then the new array is constructed as above;

            if mode='clip', values in a (and thus Ba) may be any (signed) integer; negative integers are mapped to 0;
            values greater than n-1 are mapped to n-1; and then the new array is constructed as above.

        Parameters

            choices: sequence of arrays

                Choice arrays. a and all of the choices must be broadcastable to the same shape. If choices is itself an
                 array (not recommended), then its outermost dimension (i.e., the one corresponding to choices.shape[0])
                  is taken as defining the “sequence”.

            out: array, optional

                If provided, the result will be inserted into this array. It should be of the appropriate shape and
                dtype. Note that out is always buffered if mode='raise'; use other modes for better performance.

            mode{‘raise’ (default), ‘wrap’, ‘clip’}, optional

                Specifies how indices outside [0, n-1] will be treated:

                        ‘raise’ : an exception is raised

                        ‘wrap’ : value becomes value mod n

                        ‘clip’ : values < 0 are mapped to 0, values > n-1 are mapped to n-1

        Returns
            merged_array: PhiTensor
                The merged result.

        Raises
            ValueError: shape mismatch
                If a and each choice array are not all broadcastable to the same shape.

        """
        return self._apply_self_tensor_op("choose", *args, **kwargs)
