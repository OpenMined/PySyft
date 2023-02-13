# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.ndimage import rotate

# relative
# from ....core.adp.data_subject_list import DataSubjectArray
from ....core.adp.data_subject_ledger import DataSubjectLedger
from ....core.adp.data_subject_list import DataSubject
from ...common.serde.serializable import serializable
from ...common.uid import UID

# from ..broadcastable import is_broadcastable
from ..lazy_repeat_array import lazyrepeatarray
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from .gamma_tensor import GammaTensor
from .jax_ops import SyftTerminalNoop
from .tensor_wrapped_phi_tensor_pointer import TensorWrappedPhiTensorPointer

INPLACE_OPS = {"resize", "sort"}

# def dispatch_tensor(
#     *tensors: Union[PhiTensor, GammaTensor, AcceptableSimpleType],
#     child_func: Callable,
#     min_func: Callable,
#     max_func: Callable,
#     original_func: Callable,
# ) -> Union[PhiTensor, GammaTensor]:
#     def cast_to_gamma(tensor: Union[PhiTensor, GammaTensor]) -> GammaTensor:
#         if isinstance(tensor, PhiTensor):
#             return tensor.gamma
#         return tensor

#     def check_phi_or_constant(tensor: Union[PhiTensor, GammaTensor]) -> bool:
#         return is_acceptable_simple_type(tensor) or isinstance(tensor, PhiTensor)

#     def extract_attribute_or_self(
#         tensor: Union[PhiTensor, GammaTensor], field: str
#     ) -> Union[PhiTensor, GammaTensor, DataSubject]:
#         if hasattr(tensor, field):
#             return getattr(tensor, field)
#         if field != "data_subject":
#             return tensor
#         else:
#             raise NotImplementedError

#     childs = [extract_attribute_or_self(tensor, "child") for tensor in tensors]

#     if all(map(check_phi_or_constant, tensors)):
#         min_values = [
#             extract_attribute_or_self(tensor, "min_vals") for tensor in tensors
#         ]
#         max_values = [
#             extract_attribute_or_self(tensor, "max_vals") for tensor in tensors
#         ]
#         data_subject = [
#             extract_attribute_or_self(tensor, "data_subject") for tensor in tensors
#         ]
#         data_subject = [ds for ds in data_subject if ds is not None]

#         reducer = [ds == data_subject[0] for ds in data_subject]
#         if np.all(reducer):
#             return original_func(*map(cast_to_gamma, tensors))

#         return PhiTensor(
#             child=child_func(childs),
#             min_vals=min_func(min_values),
#             max_vals=max_func(max_values),
#             data_subject=tensors[0].data_subject,  # type: ignore
#         )

#     return original_func(*map(cast_to_gamma, tensors))


@serializable(recursive_serde=True)
class PhiTensor(PassthroughTensor):
    PointerClassOverride = TensorWrappedPhiTensorPointer
    __attr_allowlist__ = ["child", "min_vals", "max_vals", "data_subject", "id"]
    __slots__ = ("child", "min_vals", "max_vals", "data_subject", "id")

    def __init__(
        self,
        child: Union[Sequence, NDArray],
        data_subject: DataSubject,
        min_vals: Union[np.ndarray, lazyrepeatarray],
        max_vals: Union[np.ndarray, lazyrepeatarray],
        id: Optional[UID] = None,
    ) -> None:
        super().__init__(child)

        # lazyrepeatarray matching the shape of child
        if not isinstance(min_vals, lazyrepeatarray):
            min_vals = lazyrepeatarray(data=min_vals, shape=child.shape)  # type: ignore
        if not isinstance(max_vals, lazyrepeatarray):
            max_vals = lazyrepeatarray(data=max_vals, shape=child.shape)  # type: ignore
        self.min_vals = min_vals
        self.max_vals = max_vals

        self.data_subject = data_subject
        if id is None:
            id = UID()
        self.id = id

    def reconstruct(self, state: Dict) -> PhiTensor:
        return state[self.id]

    @property
    def proxy_public_kwargs(self) -> Dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "data_subject": self.data_subject,
        }

    @property
    def gamma(self) -> GammaTensor:
        """Property to cast this tensor into a GammaTensor"""
        return self.create_gamma()

    def copy(self, order: Optional[str] = "K") -> PhiTensor:
        """Return copy of the given object"""

        return PhiTensor(
            child=self.child.copy(order=order),
            min_vals=self.min_vals.copy(order=order),
            max_vals=self.max_vals.copy(order=order),
            data_subject=self.data_subject,
        )

    def take(
        self,
        indices: ArrayLike,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        mode: str = "raise",
    ) -> PhiTensor:
        """Take elements from an array along an axis."""
        out_child = self.child.take(indices, axis=axis, mode=mode, out=out)
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subject=self.data_subject,
        )

    def put(
        self,
        ind: ArrayLike,
        v: ArrayLike,
        mode: str = "raise",
    ) -> PhiTensor:
        """Replaces specified elements of an array with given values.
        The indexing works on the flattened target array. put is roughly equivalent to:
            a.flat[ind] = v
        """
        if self.min_vals.data > min(v) or self.max_vals.data < max(v):
            raise Exception("The v values must be within the data bounds")

        out_child = self.child
        out_child.put(ind, v, mode=mode)
        return PhiTensor(
            child=out_child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            data_subject=self.data_subject,
        )

    def ptp(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> PhiTensor:
        out_child = self.child.ptp(axis=axis)

        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data - self.min_vals.data, shape=out_child.shape
            ),
            data_subject=self.data_subject,
        )

    def __mod__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma % other.gamma
            else:
                out_child = self.child % other.child
                return PhiTensor(
                    child=self.child % other.child,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(
                        data=min(0, other.min_vals.data), shape=out_child.shape
                    ),
                    max_vals=lazyrepeatarray(
                        data=max(0, other.max_vals.data), shape=out_child.shape
                    ),
                )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                max_vals = lazyrepeatarray(
                    data=max(0, other.max()), shape=self.child.shape
                )
                min_vals = lazyrepeatarray(
                    data=min(0, other.min()), shape=self.child.shape
                )
            else:
                max_vals = lazyrepeatarray(data=max(0, other), shape=self.child.shape)
                min_vals = lazyrepeatarray(data=min(0, other), shape=self.child.shape)

            return PhiTensor(
                child=self.child % other,
                min_vals=min_vals,
                max_vals=max_vals,
                data_subject=self.data_subject,
            )

        elif isinstance(other, GammaTensor):
            return self.gamma % other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def any(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        where: Optional[ArrayLike] = None,
    ) -> PhiTensor:
        # TODO: properly define data subjects and
        # figure out if it is not a privacy violation to return bool
        if where is None:
            out_child = np.array(self.child.any(axis=axis, keepdims=keepdims))
        else:
            out_child = np.array(
                self.child.any(axis=axis, keepdims=keepdims, where=where)
            )

        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=1, shape=out_child.shape),
            data_subject=self.data_subject,
        )

    def all(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        where: Optional[ArrayLike] = None,
    ) -> PhiTensor:
        # TODO: properly define data subjects
        if where is None:
            out_child = np.array(self.child.all(axis=axis, keepdims=keepdims))
        else:
            out_child = np.array(
                self.child.all(axis=axis, keepdims=keepdims, where=where)
            )

        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=1, shape=out_child.shape),
            data_subject=self.data_subject,
        )

    def __and__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma & other.gamma
            else:
                child = self.child & other.child
                other_min, other_max = other.min_vals.data, other.max_vals.data

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            child = self.child & other
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other

        elif isinstance(other, GammaTensor):
            return self.gamma & other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

        # TODO: should modify for a tighter found for and
        _max_vals = int(max(self.max_vals.data.max(), other_max))
        _min = int(min(self.min_vals.data.min(), other_min))
        if self.min_vals.data < 0 and other_min < 0:
            _min_vals = -(2 ** _min.bit_length())
        else:
            _min_vals = min(0, _min)

        return PhiTensor(
            child=child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=_min_vals, shape=child.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=child.shape),
        )

    def __or__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma | other.gamma
            else:
                child = self.child | other.child
                other_min, other_max = other.min_vals.data, other.max_vals.data

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
            child = self.child | other

        elif isinstance(other, GammaTensor):
            return self.gamma | other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

        # TODO: should modify for a tighter found for or
        _max = int(max(self.max_vals.data, other_max))
        _min_vals = min(self.min_vals.data, other_min)
        _max_vals = (2 ** (_max).bit_length()) - 1

        return PhiTensor(
            child=child,
            min_vals=lazyrepeatarray(data=_min_vals, shape=child.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=child.shape),
            data_subject=self.data_subject,
        )

    def copy_with(self, child: np.ndarray) -> PhiTensor:
        new_tensor = self.copy()
        new_tensor.child = child
        return new_tensor

    def __getitem__(self, item: Union[str, int, slice, PassthroughTensor]) -> PhiTensor:
        if isinstance(item, PassthroughTensor):
            data = self.child[item.child]
            return PhiTensor(
                child=data,
                min_vals=lazyrepeatarray(data=data, shape=data.shape),
                max_vals=lazyrepeatarray(data=data, shape=data.shape),
                data_subject=self.data_subject,
            )
        else:
            data = self.child[item]
            return PhiTensor(
                child=data,
                min_vals=lazyrepeatarray(data=data, shape=data.shape),
                max_vals=lazyrepeatarray(data=data, shape=data.shape),
                data_subject=self.data_subject,
            )

    def zeros_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        min_vals = self.min_vals.zeros_like(*args, **kwargs)
        max_vals = self.max_vals.zeros_like(*args, **kwargs)

        child = (
            np.zeros_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.zeros_like(*args, **kwargs)
        )

        return PhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subject=self.data_subject,
        )

    def __setitem__(
        self,
        key: Union[int, slice, NDArray],
        value: Union[PhiTensor, GammaTensor, np.ndarray],
    ) -> Union[PhiTensor, GammaTensor]:
        if isinstance(value, PhiTensor):
            self.child[key] = value.child
            minv = value.child.min()
            maxv = value.child.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

            gamma_output = self.gamma
            gamma_output[key] = value.gamma
            return gamma_output

        elif isinstance(value, GammaTensor):
            gamma = self.gamma
            gamma[key] = value
            return gamma
        elif isinstance(value, np.ndarray):
            self.child[key] = value
            minv = value.min()
            maxv = value.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

            return PhiTensor(
                child=self.child,
                data_subject=self.data_subject,
                min_vals=self.min_vals,
                max_vals=self.max_vals,
            )
        else:
            raise NotImplementedError

    def __abs__(self) -> PhiTensor:
        data = self.child
        output = np.abs(data)

        min_val = abs(self.min_vals.data)
        max_val = abs(self.max_vals.data)

        new_min_val = min(min_val, max_val)
        new_max_val = max(min_val, max_val)

        return PhiTensor(
            child=output,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=new_min_val, shape=output.shape),
            max_vals=lazyrepeatarray(data=new_max_val, shape=output.shape),
        )

    def argmax(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        child = self.child.argmax(axis=axis)
        if axis is None:
            max_value = self.child.size - 1
        else:
            max_value = np.size(self.child, axis=axis) - 1

        return PhiTensor(
            child=child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=0, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_value, shape=child.shape),
        )

    def argmin(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        child = self.child.argmin(axis=axis)
        if axis is None:
            max_value = self.child.size - 1
        else:
            max_value = np.size(self.child, axis=axis) - 1

        return PhiTensor(
            child=child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=0, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_value, shape=child.shape),
        )

    def reshape(self, *shape: Tuple[int, ...]) -> PhiTensor:
        data = self.child
        output_data = np.reshape(data, *shape)
        return PhiTensor(
            child=output_data,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape),
        )

    def pad(self, width: int, padding_mode: str = "reflect") -> PhiTensor:
        data = self.child

        if padding_mode == "reflect":
            pad_left = pad_right = pad_top = pad_bottom = width
            # RGB image
            if len(data.shape) == 3:
                output_data = np.pad(
                    data,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    padding_mode,
                )
            # Grayscale image
            elif len(data.shape) == 2:
                output_data = np.pad(
                    data, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode
                )
            else:
                output_data = np.pad(data, width, padding_mode)
        else:
            raise NotImplementedError

        output_min_val, output_max_val = output_data.min(), output_data.max()
        return PhiTensor(
            child=output_data,
            data_subject=self.data_subject,
            min_vals=output_min_val,
            max_vals=output_max_val,
        )

    def ravel(self, order: Optional[str] = "C") -> PhiTensor:
        data = self.child
        output_data = data.ravel(order=order)

        min_vals = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return PhiTensor(
            child=output_data,
            data_subject=self.data_subject,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def random_horizontal_flip(self, p: float = 0.5) -> PhiTensor:
        """Could make more efficient by not encoding/decoding FPT"""
        if np.random.random() <= p:
            return PhiTensor(
                child=np.fliplr(self.child),
                data_subject=self.data_subject,
                min_vals=self.min_vals.horizontal_flip(),
                max_vals=self.max_vals.horizontal_flip(),
            )
        else:
            return self

    def random_vertical_flip(self, p: float = 0.5) -> PhiTensor:
        """Could make more efficient by not encoding/decoding FPT"""
        if np.random.random() <= p:
            return PhiTensor(
                child=np.flipud(self.child),
                data_subject=self.data_subject,
                min_vals=self.min_vals.vertical_flip(),
                max_vals=self.max_vals.vertical_flip(),
            )
        else:
            return self

    def random_rotation(self, degrees: Union[int, Tuple]) -> PhiTensor:
        if isinstance(degrees, int):
            angle = np.random.randint(low=-degrees, high=degrees)
        elif isinstance(degrees, tuple):
            angle = np.random.randint(low=degrees[0], high=degrees[1])

        rotated_data_value = rotate(self.child, angle)

        return PhiTensor(
            child=rotated_data_value,
            data_subject=self.data_subject,
            min_vals=rotated_data_value.min(),
            max_vals=rotated_data_value.max(),
        )

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        """
        Return the maximum of an array or along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the amax method of
                sub-classes of ndarray, however any non-default value will be.
                If the sub-class’ method does not implement keepdims any exceptions will be raised.
            initial: scalar, optional
                The minimum value of an output element. Must be present to allow computation on empty slice.
                See reduce for details.

            where: array_like of bool, optional
                Elements to compare for the maximum. See reduce for details.

        Returns
            a_max: PhiTensor
                Maximum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        result = self.child.max(axis)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
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

        result = self.child.min(axis)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def _argmax(self, axis: Optional[int]) -> PhiTensor:
        return self.child.argmax(axis)

    def unravel_argmax(
        self, axis: Optional[int] = None
    ) -> Tuple[np.ndarray]:  # possible privacy violation?
        arg_result = self._argmax(axis=axis)
        shape = self.shape
        return np.unravel_index(arg_result, shape)

    def swapaxes(self, axis1: int, axis2: int) -> PhiTensor:
        """Interchange two axes of an array."""
        out_child = np.swapaxes(self.child, axis1, axis2)
        return PhiTensor(
            child=out_child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
        )

    def nonzero(self) -> PhiTensor:
        """Return the indices of the elements that are non-zero."""
        out_child = np.array(np.nonzero(self.child))
        return PhiTensor(
            child=out_child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=max(self.child.shape), shape=out_child.shape),
        )

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> PhiTensor:
        result = self.child.mean(axis, **kwargs)

        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def std(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> PhiTensor:
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

        result = self.child.std(axis, **kwargs)
        # Std is lowest when all values are the same, 0. (-ve not possible because of squaring)
        # Std is highest when half the samples are min and other half are max
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=np.array([0]), shape=result.shape),
            max_vals=lazyrepeatarray(
                data=(self.max_vals.data - self.min_vals.data) / 2, shape=result.shape
            ),
        )

    def var(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> PhiTensor:
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

        result = self.child.var(axis, **kwargs)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=0, shape=result.shape),
            max_vals=lazyrepeatarray(
                data=0.25 * (self.max_vals.data - self.min_vals.data) ** 2,
                shape=result.shape,
            ),
        )

    def sqrt(self) -> PhiTensor:
        result = np.sqrt(self.child)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(
                data=np.sqrt(self.min_vals.data), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=np.sqrt(self.max_vals.data), shape=result.shape
            ),
        )

    def normalize(
        self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]
    ) -> PhiTensor:
        # TODO: Double check if normalization bounds are correct; they might be data dependent
        if isinstance(mean, float) and isinstance(std, float):
            return PhiTensor(
                child=(self.child - mean) / std,
                data_subject=self.data_subject,
                min_vals=(self.min_vals - mean) * (1 / std),
                max_vals=(self.max_vals - mean) * (1 / std),
            )
        else:
            # This is easily doable in the future
            raise NotImplementedError

    def create_gamma(self) -> GammaTensor:
        """Return a new Gamma tensor based on this phi tensor"""
        jax_op = SyftTerminalNoop(phi_id=self.id)
        gamma_tensor = GammaTensor(
            child=self.child,
            sources={self.id: self},
            jax_op=jax_op,
            is_linear=True,
        )

        return gamma_tensor

    def view(self, *args: Any) -> PhiTensor:
        # TODO: Figure out how to fix lazyrepeatarray reshape

        data = self.child.reshape(*args)
        return PhiTensor(
            child=data,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data.min(), shape=data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data.max(), shape=data.shape),
        )

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
        private: bool,
    ) -> AcceptableSimpleType:
        gamma = self.gamma

        res = gamma.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=sigma,
            private=private,
        )
        return res

    @property
    def value(self) -> np.ndarray:
        return self.child

    def astype(self, np_type: np.dtype) -> PhiTensor:
        return self.__class__(
            child=self.child.astype(np_type),
            data_subject=self.data_subject,
            min_vals=self.min_vals.astype(np_type),
            max_vals=self.max_vals.astype(np_type),
            # scalar_manager=self.scalar_manager,
        )

    @property
    def shape(self) -> Tuple[Any, ...]:
        if self.child is None:
            return ()
        else:
            return self.child.shape

    def __repr__(self) -> str:
        """Pretty print some information, optimized for Jupyter notebook viewing."""
        return (
            f"{self.__class__.__name__}(child={self.child}, "
            + f"min_vals={self.min_vals}, max_vals={self.max_vals})"
        )

    def __eq__(self, other: Any) -> Union[PhiTensor, GammaTensor]:  # type: ignore
        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(self.child == other)
                * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                data_subject=self.data_subject,
                min_vals=lazyrepeatarray(data=0, shape=self.shape),
                max_vals=lazyrepeatarray(data=1, shape=self.shape),
            )
        elif isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma == other.gamma
            else:
                return PhiTensor(
                    child=(self.child == other.child)
                    * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
        else:
            raise NotImplementedError(
                f"__eq__ not implemented between PhiTensor and {type(other)}."
            )

    def __ne__(self, other: Any) -> Union[PhiTensor, GammaTensor]:  # type: ignore
        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(self.child != other)
                * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                data_subject=self.data_subject,
                min_vals=lazyrepeatarray(data=0, shape=self.shape),
                max_vals=lazyrepeatarray(data=1, shape=self.shape),
            )
        elif isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma != other.gamma
            else:
                return PhiTensor(
                    child=(self.child != other.child)
                    * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
        else:
            raise NotImplementedError(
                f"__ne__ not implemented between PhiTensor and {type(other)}."
            )

    def __add__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma + other.gamma

            return PhiTensor(
                child=self.child + other.child,
                min_vals=self.min_vals + other.min_vals,
                max_vals=self.max_vals + other.max_vals,
                data_subject=self.data_subject,
            )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child + other,
                min_vals=self.min_vals + other,
                max_vals=self.max_vals + other,
                data_subject=self.data_subject,
            )

        elif isinstance(other, GammaTensor):
            return self.gamma + other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError
        # try:
        # return dispatch_tensor(
        #     self,
        #     other,
        #     child_func=lambda tensors: operator.add(*tensors),
        #     min_func=lambda tensors: operator.add(*tensors),
        #     max_func=lambda tensors: operator.add(*tensors),
        #     original_func=lambda tensors: operator.add(*tensors),
        # )
        # except TypeError:
        #     raise NotImplementedError(
        #         f"__add__ not implemented for these types"
        #     )

    def __radd__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        return self.__add__(other)

    def __sub__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma - other.gamma

            return PhiTensor(
                child=self.child - other.child,
                min_vals=self.min_vals - other.min_vals,
                max_vals=self.max_vals - other.max_vals,
                data_subject=self.data_subject,
            )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child - other,
                min_vals=self.min_vals - other,
                max_vals=self.max_vals - other,
                data_subject=self.data_subject,
            )

        elif isinstance(other, GammaTensor):
            return self.gamma - other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError
        # return dispatch_tensor(
        #     self,
        #     other,
        #     child_func=lambda tensors: operator.sub(*tensors),
        #     min_func=lambda tensors: operator.sub(*tensors),
        #     max_func=lambda tensors: operator.sub(*tensors),
        #     original_func=lambda tensors: operator.sub(*tensors),
        # )

    def __rsub__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        return (self - other) * -1

    def __mul__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, PhiTensor):
            if self.data_subject == other.data_subject:
                min_min = self.min_vals.data * other.min_vals.data
                min_max = self.min_vals.data * other.max_vals.data
                max_min = self.max_vals.data * other.min_vals.data
                max_max = self.max_vals.data * other.max_vals.data

                _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

                return PhiTensor(
                    child=self.child * other.child,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                    max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                )
            else:
                return self.gamma * other.gamma

        elif is_acceptable_simple_type(other):
            data = self.child * other

            min_min = self.min_vals.data * other
            min_max = self.min_vals.data * other
            max_min = self.max_vals.data * other
            max_max = self.max_vals.data * other

            _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            min_vals = self.min_vals.copy()
            min_vals.data = _min_vals
            max_vals = self.max_vals.copy()
            max_vals.data = _max_vals

            return PhiTensor(
                child=data,
                data_subject=self.data_subject,
                min_vals=min_vals,
                max_vals=max_vals,
            )
        elif isinstance(other, GammaTensor):
            return self.gamma * other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __rmul__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma / other.gamma
            else:
                min_min = self.min_vals.data / other.min_vals.data
                min_max = self.min_vals.data / other.max_vals.data
                max_min = self.max_vals.data / other.min_vals.data
                max_max = self.max_vals.data / other.max_vals.data

                _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

                return PhiTensor(
                    child=self.child / other.child,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                    max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                )
        elif isinstance(other, GammaTensor):
            return self.gamma / other
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child / other,
                data_subject=self.data_subject,
                min_vals=lazyrepeatarray(
                    data=self.min_vals.data / other, shape=self.min_vals.shape
                ),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data / other, shape=self.max_vals.shape
                ),
            )
        else:
            raise NotImplementedError(
                f"truediv not supported between PhiTensor & {type(other)}"
            )

    def __rtruediv__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(other / self.child),
                min_vals=(other / self.min_vals),
                max_vals=(other / self.max_vals),
                data_subject=self.data_subject,
            )

        elif isinstance(other, GammaTensor):
            return (1 / self.gamma) * other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __matmul__(
        self, other: Union[np.ndarray, PhiTensor]
    ) -> Union[PhiTensor, GammaTensor]:
        if not isinstance(other, (np.ndarray, PhiTensor, GammaTensor)):
            raise Exception(
                f"Matrix multiplication not yet implemented for type {type(other)}"
            )
        else:
            if isinstance(other, np.ndarray):
                data = self.child.__matmul__(other)
                min_min = (self.min_vals @ other).data
                min_max = (self.min_vals @ other).data
                max_max = (self.max_vals @ other).data
                max_min = (self.max_vals @ other).data
                minv = np.min([min_min, min_max, max_max, max_min], axis=0)  # type: ignore
                min_vals = lazyrepeatarray(data=minv, shape=data.shape)
                max_vals = self.max_vals.__matmul__(other)
            elif isinstance(other, PhiTensor):
                if self.data_subject != other.data_subject:
                    return self.gamma @ other.gamma
                else:
                    min_min = (self.min_vals @ other.min_vals).data.min()
                    min_max = (self.min_vals @ other.max_vals).data.min()
                    max_max = (self.max_vals @ other.max_vals).data.min()
                    max_min = (self.max_vals @ other.min_vals).data.min()

                    if self.min_vals.data <= 0 and self.max_vals.data >= 0:
                        minv = np.min([0, min_min, min_max, max_max, max_min], axis=0)  # type: ignore
                    else:
                        minv = np.min([min_min, min_max, max_max, max_min], axis=0)  # type: ignore

                    data = self.child.__matmul__(other.child)
                    max_vals = self.max_vals.__matmul__(other.max_vals)
                    min_vals = lazyrepeatarray(data=minv, shape=data.shape)

            elif isinstance(other, GammaTensor):
                return self.gamma @ other
            else:
                print("Type is unsupported:" + str(type(other)))
                raise NotImplementedError

            return PhiTensor(
                child=data,
                max_vals=max_vals,
                min_vals=min_vals,
                data_subject=self.data_subject,
            )

    def argsort(self, axis: Optional[int] = -1) -> PhiTensor:
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
        result = self.child.argsort(axis)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=0, shape=self.shape),
            max_vals=lazyrepeatarray(data=self.child.size, shape=self.shape),
        )

    def sort(self, axis: int = -1, kind: Optional[str] = None) -> PhiTensor:
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

        # Must do argsort before we change self.child by calling sort
        self.child.sort(axis, kind)
        return PhiTensor(
            child=self.child,
            data_subject=self.data_subject,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
        )

    def __lshift__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other

            child = self.child << other

        elif isinstance(other, GammaTensor):
            return self.gamma << other
        elif isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma << other.gamma
            else:
                child = self.child << other.child
                other_max = other.max_vals.data
                other_min = other.min_vals.data
        else:
            raise NotImplementedError(
                f"__lshift__ not implemented between PhiTensor and {type(other)}."
            )

        min_min = self.min_vals.data << other_min
        min_max = self.min_vals.data << other_max
        max_min = self.max_vals.data << other_min
        max_max = self.max_vals.data << other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        return PhiTensor(
            child=child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
        )

    def __rshift__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
            child = self.child >> other

        elif isinstance(other, GammaTensor):
            return self.gamma >> other
        elif isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma >> other.gamma
            else:
                child = self.child >> other.child
                other_max = other.max_vals.data
                other_min = other.min_vals.data
        else:
            raise NotImplementedError(
                f"__rshift__ not implemented between PhiTensor and {type(other)}."
            )

        min_min = self.min_vals.data >> other_min
        min_max = self.min_vals.data >> other_max
        max_min = self.max_vals.data >> other_min
        max_max = self.max_vals.data >> other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        return PhiTensor(
            child=child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
        )

    def __xor__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
            child = self.child ^ other
        elif isinstance(other, GammaTensor):
            return self.gamma ^ other
        elif isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma ^ other.gamma
            else:
                child = self.child ^ other.child
                other_min, other_max = other.min_vals.data, other.max_vals.data
        else:
            raise NotImplementedError(
                f"__xor__ not implemented between PhiTensor and {type(other)}."
            )

        # TODO: should modify for a tighter found for xor
        _max = int(max(self.max_vals.data, other_max))
        _min = int(min(self.min_vals.data, other_min))
        _max_vals = max(
            (2 ** (_min ^ _max).bit_length()) - 1, (2 ** (_max).bit_length()) - 1
        )
        _min_vals = min(0, _min)

        return PhiTensor(
            child=child,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
        )

    def searchsorted(self, v: Any) -> Union[PhiTensor, GammaTensor]:
        """
        https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError

    def __divmod__(
        self, other: Any
    ) -> Tuple[Union[PhiTensor, GammaTensor], Union[PhiTensor, GammaTensor]]:
        if is_acceptable_simple_type(other) or isinstance(
            other, (PhiTensor, GammaTensor)
        ):
            return self // other, self % other  # type: ignore
        else:
            raise NotImplementedError(
                f"PhiTensor divmod not supported for type: {other}"
            )

    def divmod(
        self, other: Any
    ) -> Tuple[Union[PhiTensor, GammaTensor], Union[PhiTensor, GammaTensor]]:
        return self.__divmod__(other)

    def __round__(self, n: int = 0) -> PhiTensor:
        return PhiTensor(
            child=self.child.round(n),
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(
                data=self.min_vals.data.round(n), shape=self.min_vals.shape
            ),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data.round(n), shape=self.max_vals.shape
            ),
        )

    def round(self, n: int = 0) -> PhiTensor:
        return self.__round__(n)

    def __rmatmul__(
        self, other: Union[np.ndarray, PhiTensor]
    ) -> Union[PhiTensor, GammaTensor]:
        if not isinstance(other, (np.ndarray, PhiTensor, GammaTensor)):
            raise Exception(
                f"Matrix multiplication not yet implemented for type {type(other)}"
            )
        else:
            # Modify before merge, to know is broadcast is actually necessary
            if False:  # and not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape} and {other.shape}"
                )
            else:
                if isinstance(other, np.ndarray):
                    data = self.child.__rmatmul__(other)
                    min_vals = self.min_vals.__rmatmul__(other)
                    max_vals = self.max_vals.__rmatmul__(other)
                elif isinstance(other, PhiTensor):
                    if self.data_subject != other.data_subject:
                        return self.gamma.__rmatmul__(other.gamma)
                    else:
                        data = self.child.__rmatmul__(other.child)
                        min_vals = self.min_vals.__rmatmul__(other.min_vals)
                        max_vals = self.max_vals.__rmatmul__(other.max_vals)

                else:
                    print("Type is unsupported:" + str(type(other)))
                    raise NotImplementedError

                return PhiTensor(
                    child=data,
                    max_vals=max_vals,
                    min_vals=min_vals,
                    data_subject=self.data_subject,
                )

    def clip(self, a_min: float, a_max: float) -> PhiTensor:
        output_data = np.clip(self.child, a_min, a_max)

        min_v = np.clip(self.min_vals.data, a_min, a_max)
        max_v = np.clip(self.max_vals.data, a_min, a_max)

        min_vals = lazyrepeatarray(data=min_v, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=max_v, shape=output_data.shape)

        return PhiTensor(
            child=output_data,
            data_subject=self.data_subject,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> PhiTensor:
        """Transposes self.child, min_vals, and max_vals if these can be transposed, otherwise doesn't change them."""

        output_data = self.child.transpose(*args, **kwargs)
        min_vals = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)
        return PhiTensor(
            child=output_data,
            data_subject=self.data_subject,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def flatten(self, order: Optional[str] = "C") -> PhiTensor:
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
            order{‘C’, ‘F’, ‘A’, ‘K’}, optional
                ‘C’ means to flatten in row-major (C-style) order.
                ‘F’ means to flatten in column-major (Fortran- style) order.
                ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory,
                        row-major order otherwise.
                ‘K’ means to flatten a in the order the elements occur in memory. The default is ‘C’.
        Returns
            PhiTensor
                A copy of the input array, flattened to one dimension.
        """
        if order not in ["C", "F", "A", "K"]:
            raise NotImplementedError(f"Flatten is not implemented for order={order}")

        output_data = self.child.flatten(order=order)
        return PhiTensor(
            child=output_data,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape),
        )

    def concatenate(
        self,
        other: Union[np.ndarray, PhiTensor],
        *args: Any,
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma.concatenate(other.gamma, *args, **kwargs)

            return PhiTensor(
                child=self.child.concatenate(other.child, *args, **kwargs),
                min_vals=self.min_vals.concatenate(other.min_vals, *args, **kwargs),
                max_vals=self.max_vals.concatenate(other.max_vals, *args, **kwargs),
                data_subject=self.data_subject,
            )
        elif isinstance(other, GammaTensor):
            return self.gamma.concatenate(other, *args, **kwargs)

        elif is_acceptable_simple_type(other):
            raise NotImplementedError
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __lt__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, PhiTensor):
            if self.data_subject == other.data_subject:
                return PhiTensor(
                    child=(self.child < other.child) * 1,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__lt__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__lt__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            data = self.child < other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1

            return PhiTensor(
                child=data,
                data_subject=self.data_subject,
                min_vals=min_vals,
                max_vals=max_vals,
            )

        else:
            raise NotImplementedError

    def __le__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):
            if self.data_subject == other.data_subject:
                return PhiTensor(
                    child=(self.child <= other.child) * 1,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__le__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__le__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            data = self.child <= other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1

            return PhiTensor(
                child=data,
                data_subject=self.data_subject,
                min_vals=min_vals,
                max_vals=max_vals,
            )

        else:
            raise NotImplementedError

    def __gt__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):
            if self.data_subject == other.data_subject:
                return PhiTensor(
                    child=(self.child > other.child) * 1,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__gt__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__gt__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            data = self.child > other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1

            return PhiTensor(
                child=data,
                data_subject=self.data_subject,
                min_vals=min_vals,
                max_vals=max_vals,
            )
        else:
            raise NotImplementedError  # type: ignore

    def __ge__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):
            if self.data_subject == other.data_subject:
                return PhiTensor(
                    child=(self.child >= other.child) * 1,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__ge__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__ge__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            data = self.child >= other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1

            return PhiTensor(
                child=data,
                data_subject=self.data_subject,
                min_vals=min_vals,
                max_vals=max_vals,
            )

        else:
            return NotImplementedError  # type: ignore

    # Re enable after testing
    def dot(
        self, other: Union[PhiTensor, GammaTensor, np.ndarray]
    ) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, np.ndarray):
            return PhiTensor(
                child=np.dot(self.child, other),
                min_vals=np.dot(self.min_vals, other),
                max_vals=np.dot(self.max_vals, other),
                data_subject=self.data_subject,
            )
        elif isinstance(other, PhiTensor):
            if self.data_subject == other.data_subject:
                return PhiTensor(
                    child=np.dot(self.child, other.child),
                    min_vals=np.dot(self.min_vals, other.min_vals),
                    max_vals=np.dot(self.max_vals, other.max_vals),
                    data_subject=self.data_subject,
                )
            else:
                return self.gamma.dot(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.dot(other)
        else:
            raise NotImplementedError

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
        where: Optional[ArrayLike] = None,
    ) -> PhiTensor:
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
        if where is None:
            result = np.array(self.child.sum(axis=axis, keepdims=keepdims))
            num = np.ones_like(self.child).sum(axis=axis, keepdims=keepdims)
        else:
            result = self.child.sum(axis=axis, keepdims=keepdims, where=where)
            num = np.ones_like(self.child).sum(
                axis=axis, keepdims=keepdims, initial=initial, where=where
            )

        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data * num, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data * num, shape=result.shape),
        )

    def __pow__(
        self, power: Union[float, int], modulo: Optional[int] = None
    ) -> PhiTensor:
        if modulo is None:
            if self.min_vals.data <= 0 <= self.max_vals.data:
                # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
                minv = min(0, (self.min_vals.data**power).min())
            else:
                minv = self.min_vals.data**power

            return PhiTensor(
                child=self.child**power,
                data_subject=self.data_subject,
                min_vals=lazyrepeatarray(data=minv, shape=self.shape),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data**power, shape=self.shape
                ),
            )
        else:
            # This may be unnecessary- modulo is NotImplemented in ndarray.pow
            if self.min_vals.data <= 0 <= self.max_vals.data:
                # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
                minv = min(0, (self.min_vals.data**power).min() % modulo)
            else:
                minv = (self.min_vals.data**power) % modulo
            return PhiTensor(
                child=self.child**power % modulo,
                data_subject=self.data_subject,
                min_vals=lazyrepeatarray(data=minv, shape=self.shape),
                max_vals=lazyrepeatarray(
                    data=(self.max_vals.data**power) % modulo, shape=self.shape
                ),
            )

    def expand_dims(self, axis: int) -> PhiTensor:
        result = np.expand_dims(self.child, axis=axis)
        minv = self.min_vals.copy()
        minv.shape = result.shape
        maxv = self.max_vals.copy()
        maxv.shape = result.shape

        return PhiTensor(
            child=result,
            min_vals=minv,
            max_vals=maxv,
            data_subject=self.data_subject,
        )

    def ones_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        min_vals = self.min_vals.ones_like(*args, **kwargs)
        max_vals = self.max_vals.ones_like(*args, **kwargs)

        child = (
            np.ones_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.ones_like(*args, **kwargs)
        )

        return PhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subject=self.data_subject,
        )

    def __neg__(self) -> PhiTensor:
        return PhiTensor(
            child=self.child * -1,
            min_vals=self.max_vals * -1,
            max_vals=self.min_vals * -1,
            data_subject=self.data_subject,
        )

    def __pos__(self) -> PhiTensor:
        return PhiTensor(
            child=self.child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            data_subject=self.data_subject,
        )

    def resize(
        self, new_shape: Union[int, Tuple[int, ...]], refcheck: bool = True
    ) -> PhiTensor:
        self.child.resize(new_shape, refcheck=refcheck)
        out_shape = self.child.shape
        return PhiTensor(
            child=self.child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_shape),
            data_subject=self.data_subject,
        )

    def compress(self, condition: List[bool], axis: Optional[int] = None) -> PhiTensor:
        out_child = self.child.compress(condition, axis)
        if 0 in out_child.shape:
            raise NotImplementedError
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subject=self.data_subject,
        )

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        out_child = self.child.squeeze(axis)
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subject=self.data_subject,
        )

    def repeat(
        self, repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None
    ) -> PhiTensor:
        """
        Repeat elements of an array.

        Parameters
            repeats: int or array of ints

                The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            axis: int, optional

                The axis along which to repeat values.
                By default, use the flattened input array, and return a flat output array.

        Returns

            repeated_array: PhiTensor

                Output array which has the same shape as a, except along the given axis.

        """

        result = self.child.repeat(repeats, axis)
        if isinstance(self.min_vals, lazyrepeatarray):
            minv = lazyrepeatarray(data=self.min_vals.data.min(), shape=result.shape)
            maxv = lazyrepeatarray(data=self.max_vals.data.max(), shape=result.shape)
        else:
            minv = self.min_vals
            maxv = self.max_vals

        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=minv,
            max_vals=maxv,
        )

    def choose(
        self,
        choices: Union[Sequence, np.ndarray, PassthroughTensor],
        mode: Optional[str] = "raise",
    ) -> Union[PhiTensor, GammaTensor]:
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

        if isinstance(choices, PhiTensor):
            if self.data_subject != choices.data_subject:
                return self.gamma.choose(choices.gamma, mode=mode)
            else:
                result = self.child.choose(choices.child, mode=mode)
        elif isinstance(choices, GammaTensor):
            return self.gamma.choose(choices, mode=mode)
        else:
            raise NotImplementedError(
                f"Object type: {type(choices)} This leads to a data leak or side channel attack"
            )

        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(
                data=choices.min_vals.data.min(), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=choices.max_vals.data.max(), shape=result.shape
            ),
        )

    def cumsum(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        """
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
        result = self.child.cumsum(axis=axis)
        num = np.ones_like(self.child).cumsum(axis=axis)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(
                data=(self.min_vals.data * num).min(), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=(self.max_vals.data * num).max(), shape=result.shape
            ),
        )

    def cumprod(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
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
        result = self.child.cumprod(axis=axis)
        num = np.ones_like(self.child).cumsum(axis=axis)
        if abs(self.max_vals.data) >= (self.min_vals.data):
            highest = abs(self.max_vals.data)
        else:
            highest = abs(self.min_vals.data)

        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(
                data=-((highest**num).max()), shape=result.shape
            ),
            max_vals=lazyrepeatarray(data=(highest**num).max(), shape=result.shape),
        )

    def prod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
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
        result = self.child.prod(axis=axis)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(
                data=self.min_vals.data ** (self.child.size / result.size),
                shape=result.shape,
            ),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data ** (self.child.size / result.size),
                shape=result.shape,
            ),
        )

    def __floordiv__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        """
        return self // value.
        """
        if isinstance(other, PhiTensor):
            if self.data_subject != other.data_subject:
                return self.gamma // other.gamma
            else:
                min_min = self.min_vals.data // other.min_vals.data
                min_max = self.min_vals.data // other.max_vals.data
                max_min = self.max_vals.data // other.min_vals.data
                max_max = self.max_vals.data // other.max_vals.data

                _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

                return PhiTensor(
                    child=self.child // other.child,
                    data_subject=self.data_subject,
                    min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                    max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                )
        elif isinstance(other, GammaTensor):
            return self.gamma // other
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child // other,
                data_subject=self.data_subject,
                min_vals=lazyrepeatarray(
                    data=self.min_vals.data // other, shape=self.min_vals.shape
                ),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data // other, shape=self.max_vals.shape
                ),
            )
        else:
            raise NotImplementedError(
                f"floordiv not supported between PhiTensor & {type(other)}"
            )

    def __rfloordiv__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(other // self.child),
                min_vals=(other // self.min_vals),
                max_vals=(other // self.max_vals),
                data_subject=self.data_subject,
            )

        elif isinstance(other, GammaTensor):
            return other // self.gamma
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> PhiTensor:
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

            sum_along_diagonals: PhiTensor
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.
        """
        result = self.child.trace(offset, axis1, axis2)

        # This is potentially expensive
        num = np.ones_like(self.child).trace(offset, axis1, axis2)
        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data * num, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data * num, shape=result.shape),
        )

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> PhiTensor:
        result = self.child.diagonal(offset, axis1, axis2)

        return PhiTensor(
            child=result,
            data_subject=self.data_subject,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    # def _object2bytes(self) -> bytes:
    #     schema = get_capnp_schema(schema_file="phi_tensor.capnp")

    #     pt_struct: CapnpModule = schema.PT  # type: ignore
    #     pt_msg = pt_struct.new_message()
    #     # this is how we dispatch correct deserialization of bytes
    #     pt_msg.magicHeader = serde_magic_header(type(self))

    #     if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
    #         chunk_bytes(capnp_serialize(np.array(self.child), to_bytes=True), "child", pt_msg)  # type: ignore
    #         pt_msg.isNumpy = True
    #     else:
    #         chunk_bytes(serialize(self.child, to_bytes=True), "child", pt_msg)  # type: ignore
    #         pt_msg.isNumpy = False

    #     pt_msg.minVals = serialize(self.min_vals, to_bytes=True)
    #     pt_msg.maxVals = serialize(self.max_vals, to_bytes=True)
    #     pt_msg.dataSubject = serialize(dstonumpyutf8(self.data_subject), to_bytes=True)
    #     pt_msg.id = self.id.to_string()
    #     # to pack or not to pack?
    #     # to_bytes = pt_msg.to_bytes()

    #     return pt_msg.to_bytes_packed()

    # @staticmethod
    # def _bytes2object(buf: bytes) -> PhiTensor:
    #     schema = get_capnp_schema(schema_file="phi_tensor.capnp")
    #     pt_struct: CapnpModule = schema.PT  # type: ignore
    #     # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
    #     MAX_TRAVERSAL_LIMIT = 2**64 - 1
    #     # to pack or not to pack?
    #     # pt_msg = pt_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
    #     pt_msg = pt_struct.from_bytes_packed(
    #         buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    #     )

    #     if pt_msg.isNumpy:
    #         child = capnp_deserialize(combine_bytes(pt_msg.child), from_bytes=True)
    #     else:
    #         child = deserialize(combine_bytes(pt_msg.child), from_bytes=True)

    #     min_vals = deserialize(pt_msg.minVals, from_bytes=True)
    #     max_vals = deserialize(pt_msg.maxVals, from_bytes=True)
    #     data_subject = numpyutf8tods(deserialize(pt_msg.dataSubject, from_bytes=True))
    #     id_str = UID.from_string(pt_msg.id)
    #     return PhiTensor(
    #         child=child,
    #         min_vals=min_vals,
    #         max_vals=max_vals,
    #         data_subject=data_subject,
    #         id=id_str,
    #     )
