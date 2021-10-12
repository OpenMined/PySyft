# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Sequence

# third party
from nacl.signing import VerifyKey
import numpy as np
from sympy.ntheory.factor_ import factorint

# relative
from ....core.adp.entity import DataSubjectGroup
from ....core.adp.entity import Entity
from ...adp.publish import publish
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.serializable import serializable
from ...tensor.passthrough import PassthroughTensor  # type: ignore
from ...tensor.passthrough import is_acceptable_simple_type  # type: ignore
from ..broadcastable import is_broadcastable
from .adp_tensor import ADPTensor

SupportedChainType = Union[int, bool, float, np.ndarray, PassthroughTensor]


@serializable(recursive_serde=True)
class IntermediateGammaTensor(PassthroughTensor, ADPTensor):

    """Functionality for tracking differential privacy when individual values
    are contributed to by multiple entities. IntermediateGammaTensor differs
    from IniitalGammaTensor only in that InitialGammaTensor has additional
    functionality in its constructor essential to when one initially begins
    tracking metadata across mutliple entities, whereas IntermediateGammaTensor
    has a simpler constructor for use when performing operations across one or
    more IntermediateGammaTensor objects.
    """

    __attr_allowlist__ = [
        "term_tensor",
        "coeff_tensor",
        "bias_tensor",
        "scalar_manager",
        "child",
        "unique_entities",
        "n_entities",
    ]

    def __init__(
        self,
        term_tensor: np.ndarray,
        coeff_tensor: np.ndarray,
        bias_tensor: np.ndarray,
        # min_vals: np.ndarray,
        # max_vals: np.ndarray,
        scalar_manager: VirtualMachinePrivateScalarManager = VirtualMachinePrivateScalarManager(),
    ) -> None:
        super().__init__(term_tensor)

        # EXPLAIN A: if our clipped polynomial is y = clip(mx + b, min=min_vals, max=max_vals)
        # EXPLAIN B: if self.child = 5x10

        # EXPLAIN A: this is "x"
        # EXPLAIN B: this is a 5x10x1
        self.term_tensor = term_tensor

        # EXPLAIN A: this is "m"
        # EXPLAIN B: this is a 5x10x1
        self.coeff_tensor = coeff_tensor

        # EXPLAIN A: this is "b"
        # EXPLAIN B: this is a 5x10
        self.bias_tensor = bias_tensor

        # EXPLAIN A: this is "min_vals"
        # EXPLAIN B: this is a 5x10
        # self.min_vals = min_vals

        # EXPLAIN A: this is "max_vals"
        # EXPLAIN B: this is a 5x10
        # self.max_vals = max_vals

        self.scalar_manager = scalar_manager

        # Unique entities
        self.unique_entities: set[Entity] = set()
        self.n_entities = 0

        for entity in set(self._entities(to_array=False)):
            if isinstance(entity, Entity):
                if entity not in self.unique_entities:
                    self.unique_entities.add(entity)
                    self.n_entities += 1
            elif isinstance(entity, DataSubjectGroup):
                for e in entity.entity_set:
                    if e not in self.unique_entities:
                        self.unique_entities.add(e)
                        self.n_entities += 1
            else:
                raise Exception(f"{type(entity)}")

    @property
    def flat_scalars(self) -> List[Any]:
        flattened_terms = self.term_tensor.reshape(-1, self.term_tensor.shape[-1])
        flattened_coeffs = self.coeff_tensor.reshape(-1, self.coeff_tensor.shape[-1])
        flattened_bias = self.bias_tensor.reshape(-1)
        # flattened_min_vals = self.min_vals.reshape(-1)
        # flattened_max_vals = self.max_vals.reshape(-1)

        scalars = list()

        for i in range(len(flattened_terms)):
            single_poly_terms = flattened_terms[i]
            single_poly_coeffs = flattened_coeffs[i]
            single_poly_bias = flattened_bias[i]
            # single_poly_min_val = flattened_min_vals[i]
            # single_poly_max_val = flattened_max_vals[i]

            scalar = single_poly_bias

            for j in range(len(single_poly_terms)):
                term = single_poly_terms[j]
                coeff = single_poly_coeffs[j]

                for prime, n_times in factorint(term).items():
                    input_scalar = self.scalar_manager.prime2symbol[prime]
                    right = input_scalar * n_times * coeff
                    scalar = scalar + right

            scalars.append(scalar)

        return scalars

    def _values(self) -> np.array:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """
        return np.array(list(map(lambda x: x.value, self.flat_scalars))).reshape(
            self.shape
        )

    def _entities(self, to_array: bool = True) -> Union[np.array, list]:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """

        """WARNING/PLEA: DO NOT DELETE ANY OF THE COMMENTED PARTS- WE MIGHT REVERT BACK TO THEM LATER"""
        output_entities = []
        for flat_scalar in self.flat_scalars:
            # TODO: This will fail if the nested entity is any deeper than 2 levels- i.e. [A, [A, [A, B]]]. Recursive?
            combined_entities = DataSubjectGroup()
            for row in flat_scalar.input_entities:
                if isinstance(row, Entity) or isinstance(row, DataSubjectGroup):
                    combined_entities += row
                elif isinstance(row, list):
                    for i in row:
                        if isinstance(i, Entity) or isinstance(i, DataSubjectGroup):
                            combined_entities += i
                        else:
                            raise Exception(f"Not implemented for i of type:{type(i)}")
                else:
                    raise Exception(f"No plans for row type:{type(row)}")
            output_entities.append(combined_entities)
        if to_array:
            return np.array(output_entities).reshape(self.shape)
        elif not to_array:
            return output_entities
        else:
            raise Exception(f"{to_array}")

    def __gt__(self, other: Union[np.ndarray, IntermediateGammaTensor]) -> Any:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=vals > other,
                    min_vals=np.zeros_like(vals),
                    max_vals=np.ones_like(vals),
                    entities=self._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        elif isinstance(other, IntermediateGammaTensor):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                self_vals = self._values()
                other_vals = other._values()
                tensor = InitialGammaTensor(
                    values=self_vals > other_vals,
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return tensor

    def __lt__(self, other: Union[np.ndarray, IntermediateGammaTensor]) -> Any:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=vals < other,
                    min_vals=np.zeros_like(vals),
                    max_vals=np.ones_like(vals),
                    entities=self._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        elif isinstance(other, IntermediateGammaTensor):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                self_vals = self._values()
                other_vals = other._values()
                tensor = InitialGammaTensor(
                    values=self_vals < other_vals,
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return tensor

    def __eq__(self, other: Union[np.ndarray, IntermediateGammaTensor]) -> Any:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=not (vals < other) and not (vals > other),
                    max_vals=np.ones_like(vals),
                    min_vals=np.zeros_like(vals),
                    entities=self._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        elif isinstance(other, IntermediateGammaTensor):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                self_vals = self._values()
                other_vals = other._values()
                tensor = InitialGammaTensor(
                    values=self_vals == other_vals,
                    # values= not (self_vals < other_vals) and not (self_vals > other_vals),
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return tensor

    def __ne__(self, other: Union[np.ndarray, IntermediateGammaTensor]) -> Any:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=(vals < other) or (vals > other),
                    max_vals=np.ones_like(vals),
                    min_vals=np.zeros_like(vals),
                    entities=self._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        elif isinstance(other, IntermediateGammaTensor):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                self_vals = self._values()
                other_vals = other._values()
                tensor = InitialGammaTensor(
                    values=self_vals != other_vals,
                    # values= not (self_vals < other_vals) and not (self_vals > other_vals),
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return tensor

    def __ge__(self, other: Union[np.ndarray, IntermediateGammaTensor]) -> Any:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=vals >= other,
                    max_vals=np.ones_like(vals),
                    min_vals=np.zeros_like(vals),
                    entities=self._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        elif isinstance(other, IntermediateGammaTensor):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                self_vals = self._values()
                other_vals = other._values()
                tensor = InitialGammaTensor(
                    values=self_vals >= other_vals,
                    # values= not (self_vals < other_vals) and not (self_vals > other_vals),
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return tensor

    def __le__(self, other: Union[np.ndarray, IntermediateGammaTensor]) -> Any:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=vals <= other,
                    max_vals=np.ones_like(vals),
                    min_vals=np.zeros_like(vals),
                    entities=self._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        elif isinstance(other, IntermediateGammaTensor):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                self_vals = self._values()
                other_vals = other._values()
                tensor = InitialGammaTensor(
                    values=self_vals <= other_vals,
                    # values= not (self_vals < other_vals) and not (self_vals > other_vals),
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                )
            else:
                raise Exception(
                    f"Tensor shapes not compatible: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.term_tensor.shape[:-1]

    @property
    def full_shape(self) -> Tuple[int, ...]:
        return self.term_tensor.shape

    def publish(self, acc: Any, sigma: float, user_key: VerifyKey) -> np.ndarray:

        result = np.array(
            publish(
                scalars=self.flat_scalars,
                acc=acc,
                sigma=sigma,
                user_key=user_key,
                public_only=True,
            )
        ).reshape(self.shape)

        if self.sharetensor_values is not None:
            # relative
            from ..smpc.share_tensor import ShareTensor

            result = ShareTensor(
                rank=self.sharetensor_values.rank,
                nr_parties=self.sharetensor_values.nr_parties,
                ring_size=self.sharetensor_values.ring_size,
                value=result,
            )
        return result

    def sum(self, axis: Optional[int] = None) -> IntermediateGammaTensor:

        new_term_tensor = np.swapaxes(self.term_tensor, axis, -1).squeeze(axis)  # type: ignore
        new_coeff_tensor = np.swapaxes(self.coeff_tensor, axis, -1).squeeze(axis)  # type: ignore
        new_bias_tensor = self.bias_tensor.sum(axis)

        return IntermediateGammaTensor(
            term_tensor=new_term_tensor,
            coeff_tensor=new_coeff_tensor,
            bias_tensor=new_bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    def prod(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> IntermediateGammaTensor:
        new_term_tensor = self.term_tensor.prod(axis)
        new_coeff_tensor = self.coeff_tensor.prod(axis)
        new_bias_tensor = self.bias_tensor.prod(axis)
        return IntermediateGammaTensor(
            term_tensor=new_term_tensor,
            coeff_tensor=new_coeff_tensor,
            bias_tensor=new_bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    def __add__(self, other: Any) -> IntermediateGammaTensor:

        if is_acceptable_simple_type(other):

            # EXPLAIN A: if our polynomail is y = mx + b
            # EXPLAIN B: if self.child = 5x10

            # EXPLAIN A: this is "x"
            # EXPLAIN B: this is a 5x10x1
            term_tensor = self.term_tensor

            # EXPLAIN A: this is "m"
            # EXPLAIN B: this is a 5x10x1
            coeff_tensor = self.coeff_tensor

            # EXPLAIN A: this is "b"
            # EXPLAIN B: this is a 5x10
            bias_tensor = self.bias_tensor + other

        else:

            if self.scalar_manager != other.scalar_manager:
                # TODO: come up with a method for combining symbol factories
                raise Exception(
                    "Cannot add two tensors with different symbol encodings"
                )

            # Step 1: Concatenate
            term_tensor = np.concatenate([self.term_tensor, other.term_tensor], axis=-1)  # type: ignore
            coeff_tensor = np.concatenate(  # type: ignore
                [self.coeff_tensor, other.coeff_tensor], axis=-1
            )
            bias_tensor = self.bias_tensor + other.bias_tensor

        # EXPLAIN B: NEW OUTPUT becomes a 5x10x2
        # TODO: Step 2: Reduce dimensionality if possible (look for duplicates)
        return IntermediateGammaTensor(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    # def clip(self, a_min:int, a_max: int) -> IntermediateGammaTensor:
    #     """Clips the tensor at a certain minimum and maximum. a_min and a_max are
    #     assumed to be integers at present because IntermediateGammaTensor only
    #     operates over integer values at present."""
    #
    #     return None

    def __sub__(self, other: Any) -> IntermediateGammaTensor:

        if is_acceptable_simple_type(other):

            # EXPLAIN A: if our polynomail is y = mx + b
            # EXPLAIN B: if self.child = 5x10

            # EXPLAIN A: this is "x"
            # EXPLAIN B: this is a 5x10x1
            term_tensor = self.term_tensor

            # EXPLAIN A: this is "m"
            # EXPLAIN B: this is a 5x10x1
            coeff_tensor = self.coeff_tensor

            # EXPLAIN A: this is "b"
            # EXPLAIN B: this is a 5x10
            bias_tensor = self.bias_tensor - other

        else:
            if self.scalar_manager != other.scalar_manager:
                # TODO: come up with a method for combining symbol factories
                raise Exception(
                    "Cannot add two tensors with different symbol encodings"
                )

            # Step 1: Concatenate
            term_tensor = np.concatenate([self.term_tensor, other.term_tensor], axis=-1)  # type: ignore
            coeff_tensor = np.concatenate(  # type: ignore
                [self.coeff_tensor, other.coeff_tensor * -1], axis=-1
            )
            bias_tensor = self.bias_tensor - other.bias_tensor

        # EXPLAIN B: NEW OUTPUT becomes a 5x10x2
        # TODO: Step 2: Reduce dimensionality if possible (look for duplicates)
        return IntermediateGammaTensor(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    def repeat(
        self, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> IntermediateGammaTensor:
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor.repeat(*args, **kwargs).reshape(
                -1, self.term_tensor.shape[-1]
            ),
            coeff_tensor=self.coeff_tensor.repeat(*args, **kwargs).reshape(
                -1, self.coeff_tensor.shape[-1]
            ),
            bias_tensor=self.bias_tensor.repeat(*args, **kwargs),
            scalar_manager=self.scalar_manager,
        )

    def __mul__(self, other: Any) -> IntermediateGammaTensor:

        # EXPLAIN A: if our polynomial is y = mx
        # EXPLAIN B: self.child = 10x5

        if is_acceptable_simple_type(other):

            # this is "x"
            # this is 10x5x1
            term_tensor = self.term_tensor

            # term_tensor is prime because if I have variable "3" and variable "5" then
            # then variable "3 * 5 = 15" is CERTAIN to be the multiplication of ONLY "3" and "5"

            # this is "m"
            coeff_tensor = self.coeff_tensor * other

            bias_tensor = self.bias_tensor * other

        else:
            if self.scalar_manager != other.scalar_manager:
                # TODO: come up with a method for combining symbol factories
                raise Exception(
                    "Cannot add two tensors with different symbol encodings"
                )

            terms = list()
            for self_dim in range(self.term_tensor.shape[-1]):
                for other_dim in range(other.term_tensor.shape[-1]):
                    new_term = np.expand_dims(  # type: ignore
                        self.term_tensor[..., self_dim]
                        * other.term_tensor[..., other_dim],
                        -1,
                    )
                    terms.append(new_term)

            for self_dim in range(self.term_tensor.shape[-1]):
                new_term = np.expand_dims(self.term_tensor[..., self_dim], -1)  # type: ignore
                terms.append(new_term)

            for other_dim in range(self.term_tensor.shape[-1]):
                new_term = np.expand_dims(other.term_tensor[..., self_dim], -1)  # type: ignore
                terms.append(new_term)

            term_tensor = np.concatenate(terms, axis=-1)  # type: ignore

            coeffs = list()
            for self_dim in range(self.coeff_tensor.shape[-1]):
                for other_dim in range(other.coeff_tensor.shape[-1]):
                    new_coeff = np.expand_dims(  # type: ignore
                        self.coeff_tensor[..., self_dim]
                        * other.coeff_tensor[..., other_dim],
                        -1,
                    )
                    coeffs.append(new_coeff)

            for self_dim in range(self.coeff_tensor.shape[-1]):
                new_coeff = np.expand_dims(  # type: ignore
                    self.coeff_tensor[..., self_dim] * other.bias_tensor, -1
                )
                coeffs.append(new_coeff)

            for other_dim in range(self.coeff_tensor.shape[-1]):
                new_coeff = np.expand_dims(  # type: ignore
                    other.coeff_tensor[..., self_dim] * self.bias_tensor, -1
                )
                coeffs.append(new_coeff)

            coeff_tensor = np.concatenate(coeffs, axis=-1)  # type: ignore

            bias_tensor = self.bias_tensor * other.bias_tensor

        # TODO: Step 2: Reduce dimensionality if possible (look for duplicates)
        return IntermediateGammaTensor(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    def __pos__(self) -> IntermediateGammaTensor:
        return self

    def __neg__(self) -> IntermediateGammaTensor:
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor,
            coeff_tensor=-self.coeff_tensor,
            bias_tensor=-self.bias_tensor,
            scalar_manager=self.scalar_manager
        )

    def copy(self) -> IntermediateGammaTensor:
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor,
            coeff_tensor=self.coeff_tensor,
            bias_tensor=self.bias_tensor,
            scalar_manager=self.scalar_manager
        )

    def transpose(self, axes: Optional[Union[int, Sequence[int], Tuple[int]]] = None) -> IntermediateGammaTensor:
        # TODO: Need to check if new prime numbers are issued or if old ones are just moved around.
        # TODO: Need to check what's going wrong with _values() after transposing
        num = len(self.shape)
        if not axes:
            axes = [i for i in range(num)][::-1] + [num]  # Shape of last axis mustn't change
        else:
            if isinstance(axes, list):
                axes += [num]
            elif isinstance(axes, tuple):
                axes = list(axes) + [num]
            else:
                raise Exception(
                    f"Unknown type: {type(axes)}"
                )
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor.transpose(axes),
            coeff_tensor=self.term_tensor.transpose(axes),
            bias_tensor=self.bias_tensor.transpose(axes[:-1]),
            scalar_manager=self.scalar_manager
        )

    def reshape(self, *dims: Sequence[int]) -> IntermediateGammaTensor:
        # The last axis isn't visible to the user and doesn't change shape
        immutable = [self.shape[-1]]
        if isinstance(dims, tuple):
            dims = list(dims) + immutable
        elif isinstance(dims, list):
            dims += immutable
        else:
            raise Exception(
                f"Unknown type: {type(dims)}"
            )
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor.transpose(*dims),
            coeff_tensor=self.term_tensor.transpose(*dims),
            bias_tensor=self.bias_tensor.transpose(*dims[:-1]),
            scalar_manager=self.scalar_manager
        )

    def resize(
        self,
        new_shape: Union[int, Tuple[int, ...]],
        refcheck: Optional[bool] = True,
    ) -> None:
        if isinstance(new_shape, tuple):
            new_shape = list(new_shape) + [self.shape[-1]]
        self.term_tensor.resize(new_shape)
        self.coeff_tensor.resize(new_shape)
        self.bias_tensor.resize(new_shape[:-1])

    def ravel(self, order: Optional[str] = "C") -> IntermediateGammaTensor:
        # TODO: Check effect of ravel on higher dimensional arrays
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor.ravel(order),
            coeff_tensor=self.coeff_tensor.ravel(order),
            bias_tensor=self.bias_tensor.ravel(order),
            scalar_manager=self.scalar_manager
        )

    def squeeze(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> PassthroughTensor:
        pass

    def swapaxes(self, axis1: int, axis2: int) -> PassthroughTensor:
        pass

    def partition(
        self,
        kth: Union[int, Tuple[int, ...]],
        axis: Optional[int] = -1,
        kind: Optional[str] = "introselect",
        order: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> PassthroughTensor:
        pass

    def compress(
        self, condition: List[bool], axis: int = None, out: Optional[np.ndarray] = None
    ) -> PassthroughTensor:
        pass

    def __and__(self, other):
        pass

    def __or__(self, other):
        pass

    def take(
        self, indices: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> PassthroughTensor:
        pass

    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> IntermediateGammaTensor:
        last_dim = len(self.shape) - 1
        if axis1 == -1 or axis1 == last_dim or axis2 == -1 or axis2 == last_dim:
            raise Exception

        pass



