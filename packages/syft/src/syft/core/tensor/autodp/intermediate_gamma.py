# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np
from sympy.ntheory.factor_ import factorint

# relative
from ....core.adp.entity import DataSubjectGroup
from ....core.adp.entity import Entity
from ....core.adp.scalar.intermediate_gamma_scalar import IntermediateGammaScalar
from ....util import concurrency_count
from ....util import list_sum
from ....util import parallel_execution
from ....util import split_rows
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
        "value_tensor",
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
        value_tensor: Optional[np.ndarray] = None,
        # min_vals: np.ndarray,
        # max_vals: np.ndarray,
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,
        unique_entities: Optional[Set[Entity]] = None,
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

        # optional cache of values
        self.value_tensor = value_tensor

        # EXPLAIN A: this is "min_vals"
        # EXPLAIN B: this is a 5x10
        # self.min_vals = min_vals

        # EXPLAIN A: this is "max_vals"
        # EXPLAIN B: this is a 5x10
        # self.max_vals = max_vals

        self.scalar_manager = (
            scalar_manager
            if scalar_manager is not None
            else VirtualMachinePrivateScalarManager()
        )

        if not hasattr(self, "_min_vals_cache"):
            self._min_vals_cache: Optional[np.array] = None
        if not hasattr(self, "_max_vals_cache"):
            self._max_vals_cache: Optional[np.array] = None

        self.unique_entities: set[Entity] = (
            unique_entities if unique_entities is not None else set()
        )
        self.n_entities: int = len(self.unique_entities)

        if self.n_entities == 0:
            # Unique entities

            for entity in set(self._entities_list()):
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
        # flattened_min_vals = self._min_values().reshape(-1)
        # flattened_max_vals = self._max_values().reshape(-1)

        known_primes: set[int] = set(self.scalar_manager.prime2symbol.keys())

        scalars = list()
        for i in range(len(flattened_terms)):

            single_poly_terms = flattened_terms[i]
            single_poly_coeffs = flattened_coeffs[i]
            single_poly_bias = flattened_bias[i]
            # single_poly_min_val = flattened_min_vals[i]
            # single_poly_max_val = flattened_max_vals[i]

            input_mp = [single_poly_bias]

            for j in range(len(single_poly_terms)):
                term = single_poly_terms[j]
                coeff = single_poly_coeffs[j]

                if term in known_primes:
                    prime_list = [(term, 1)]
                else:
                    prime_list = factorint(term).items()

                for prime, n_times in prime_list:
                    input_scalar = self.scalar_manager.prime2symbol[int(prime)]
                    right = input_scalar * n_times * coeff
                    input_mp.append(right)

            num_process = min(concurrency_count(), len(input_mp))
            if num_process > 1:
                args = split_rows(input_mp, cpu_count=num_process)
                output = parallel_execution(list_sum, cpu_bound=True)(args)
                scalar: IntermediateGammaScalar = sum(output)  # type: ignore
            else:
                scalar: IntermediateGammaScalar = sum(input_mp)  # type: ignore

            # to optimize down stream we can prevent search on linear queries if we
            # know that all single_poly_terms are prime therefore the query is linear
            scalar.is_linear = True
            if j in single_poly_terms:
                if (
                    single_poly_terms[j] not in known_primes
                    or single_poly_coeffs[j] != 1
                ):
                    scalar.is_linear = False
                    break

            scalars.append(scalar)

        if self.value_tensor is not None:
            flat_value_tensor = self.value_tensor.flatten()
            for i in range(len(scalars)):
                scalars[i]._value_cache = flat_value_tensor[i]  # type: ignore

        return scalars

    def update(self) -> None:
        # TODO: Recalculate term tensor using an optimized slice method

        # index_number = self._entities()  # TODO: Check if this works!!
        prime_numbers = np.array(self.scalar_manager.primes_allocated, dtype=np.int32)

        # TODO: See if this fails for addition of more than 2 IGTs
        length = len(prime_numbers)
        tensor_shape_count = np.prod(self.term_tensor.shape)
        if length == tensor_shape_count:
            self.term_tensor = prime_numbers.reshape(self.term_tensor.shape)
        elif length > tensor_shape_count:
            self.term_tensor = prime_numbers[-tensor_shape_count:].reshape(
                self.term_tensor.shape
            )
        else:
            raise Exception(f"Failed to update IGT with {length} {tensor_shape_count}")

        # self.term_tensor.flatten().reshape(-1, 2)[:, -1] = prime_numbers

    def _values(self) -> np.array:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """
        return np.array(list(map(lambda x: x.value, self.flat_scalars))).reshape(
            self.shape
        )

    def _max_values(self) -> np.array:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """
        if self._max_vals_cache is not None:
            return self._max_vals_cache

        return np.array(list(map(lambda x: x.max_val, self.flat_scalars))).reshape(
            self.shape
        )

    def _min_values(self) -> np.array:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """

        if self._min_vals_cache is not None:
            return self._min_vals_cache

        return np.array(list(map(lambda x: x.min_val, self.flat_scalars))).reshape(
            self.shape
        )

    def _entities_list(self) -> list:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """

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
        return output_entities

    def _entities(self) -> np.array:
        """WARNING: DO NOT MAKE THIS AVAILABLE TO THE POINTER!!!
        DO NOT ADD THIS METHOD TO THE AST!!!
        """

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
        return np.array(output_entities).reshape(self.shape)

    def astype(self, np_type: np.dtype = np.int32) -> IntermediateGammaTensor:
        return self.__class__(
            term_tensor=self.term_tensor.astype(np_type),
            coeff_tensor=self.coeff_tensor.astype(np_type),
            bias_tensor=self.bias_tensor.astype(np_type),
            scalar_manager=self.scalar_manager,
            unique_entities=self.unique_entities,
        )

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
                    values=vals == other,
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
                # output_scalar = self.scalar_manager.copy()
                # output_scalar.combine_(other.scalar_manager)
                # other.update()
                tensor = InitialGammaTensor(
                    values=self_vals == other_vals,
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                    # scalar_manager=output_scalar
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
                    values=vals != other,
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
                # output_scalar = self.scalar_manager.copy()
                # output_scalar.combine_(other.scalar_manager)
                # other.update()
                tensor = InitialGammaTensor(
                    values=self_vals != other_vals,
                    # values= not (self_vals < other_vals) and not (self_vals > other_vals),
                    min_vals=np.zeros_like(self_vals),
                    max_vals=np.ones_like(self_vals),
                    entities=self._entities() + other._entities(),
                    # scalar_manager=output_scalar
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
        print("IntermediaGammaTensor:510: TRY: publish(scalars=self.flat_scalars)")
        result = np.array(
            publish(
                scalars=self.flat_scalars,
                acc=acc,
                sigma=sigma,
                user_key=user_key,
                public_only=True,
            )
        ).reshape(self.shape)
        print("IntermediaGammaTensor:510: SUCCESS: publish(scalars=self.flat_scalars)")
        sharetensor_values = getattr(self, "sharetensor_values", None)
        if sharetensor_values is not None:
            # relative
            from ..smpc.share_tensor import ShareTensor

            result = ShareTensor(
                rank=sharetensor_values.rank,
                parties_info=sharetensor_values.parties_info,
                ring_size=sharetensor_values.ring_size,
                seed_przs=sharetensor_values.seed_przs,
                clients=sharetensor_values.clients,
                value=result,
            )
        return result

    def sum(self, axis: Optional[int] = None) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().sum(axis),
            entities=self._entities().sum(axis),
            max_vals=self._max_values().sum(axis),
            min_vals=self._min_values().sum(axis),
        )

    def prod(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().prod(axis),
            entities=self._entities().sum(
                axis
            ),  # Entities get added (combined) instead of multiplied
            max_vals=self._max_values().prod(axis),
            min_vals=self._min_values().prod(axis),
        )

    def __add__(self, other: Any) -> IntermediateGammaTensor:
        output_scalar_manager = self.scalar_manager.copy()
        # TODO: add support for SingleEntitiyPhiTensor
        # this will cause it to generate them using a more computationally intensive
        unique_entities = None

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

            max_vals_cache = self._max_values() + other
            min_vals_cache = self._min_values() + other
            unique_entities = self.unique_entities

        else:

            # relative
            from .dp_tensor_converter import convert_to_gamma_tensor
            from .single_entity_phi import SingleEntityPhiTensor

            if isinstance(other, SingleEntityPhiTensor):
                other = convert_to_gamma_tensor(other)

            # if self.scalar_manager != other.scalar_manager:
            # TODO: come up with a method for combining symbol factories
            # raise Exception(
            #     "Cannot add two tensors with different symbol encodings"
            # )

            output_scalar_manager.combine_(other.scalar_manager)
            other.update()  # change term tensor after combining scalar managers

            # Step 1: Concatenate
            term_tensor = np.concatenate([self.term_tensor, other.term_tensor], axis=-1)  # type: ignore
            coeff_tensor = np.concatenate(  # type: ignore
                [self.coeff_tensor, other.coeff_tensor], axis=-1
            )
            bias_tensor = self.bias_tensor + other.bias_tensor

            max_vals_cache = self._max_values() + other._max_values()
            min_vals_cache = self._min_values() + other._min_values()

            if hasattr(other, "unique_entities"):
                unique_entities = self.unique_entities.union(other.unique_entities)

        # EXPLAIN B: NEW OUTPUT becomes a 5x10x2
        # TODO: Step 2: Reduce dimensionality if possible (look for duplicates)
        result = IntermediateGammaTensor(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=output_scalar_manager,
            unique_entities=unique_entities,
        )

        result._min_vals_cache = min_vals_cache
        result._max_vals_cache = max_vals_cache

        return result

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
            # relative
            from .dp_tensor_converter import convert_to_gamma_tensor
            from .single_entity_phi import SingleEntityPhiTensor

            if isinstance(other, SingleEntityPhiTensor):
                other = convert_to_gamma_tensor(other)

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

            for _ in range(self.term_tensor.shape[-1]):
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

            for _ in range(self.coeff_tensor.shape[-1]):
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

    def __matmul__(self, other: Any) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        self_values = self._values()
        self_entities = self._entities()
        self_min = self._min_values()
        self_max = self._max_values()

        # Private-Public
        if isinstance(other, np.ndarray):
            return InitialGammaTensor(
                values=self_values.__matmul__(other),
                entities=self_entities.__matmul__(other),
                min_vals=self_min.__matmul__(other),
                max_vals=self_max.__matmul__(other),
            )
        else:  # Private-Private
            # relative
            from .single_entity_phi import SingleEntityPhiTensor

            if isinstance(other, SingleEntityPhiTensor):
                # relative
                from .dp_tensor_converter import convert_to_gamma_tensor

                other_gamma = convert_to_gamma_tensor(other)
                return InitialGammaTensor(
                    values=self_values.__matmul__(other_gamma._values()),
                    entities=self_entities.__matmul__(other_gamma._entities()),
                    min_vals=self_min.__matmul__(other_gamma._min_values()),
                    max_vals=self_max.__matmul__(other_gamma._max_values()),
                )
            elif isinstance(other, IntermediateGammaTensor):
                return InitialGammaTensor(
                    values=self_values.__matmul__(other._values()),
                    entities=self_entities.__matmul__(other._entities()),
                    min_vals=self_min.__matmul__(other._min_values()),
                    max_vals=self_max.__matmul__(other._max_values()),
                )
            raise NotImplementedError

    def __pos__(self) -> IntermediateGammaTensor:
        return self

    def __neg__(self) -> IntermediateGammaTensor:
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor,
            coeff_tensor=-self.coeff_tensor,
            bias_tensor=-self.bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    def copy(self) -> IntermediateGammaTensor:
        return IntermediateGammaTensor(
            term_tensor=self.term_tensor,
            coeff_tensor=self.coeff_tensor,
            bias_tensor=self.bias_tensor,
            scalar_manager=self.scalar_manager,
        )

    def flatten(self, order: Optional[str] = "C") -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().flatten(order),
            entities=self._entities().flatten(order),
            min_vals=self._min_values().flatten(order),
            max_vals=self._max_values().flatten(order),
            scalar_manager=self.scalar_manager,
        )

    def transpose(
        self, axes: Optional[Union[int, Sequence[int], Tuple[int]]] = None
    ) -> IntermediateGammaTensor:
        # TODO: Need to check if new prime numbers are issued or if old ones are just moved around.
        # relative
        from .initial_gamma import InitialGammaTensor

        output_values = self._values().transpose(axes)
        shape = output_values.shape

        return InitialGammaTensor(
            values=output_values,
            entities=self._entities().transpose(axes),
            min_vals=self._min_values().reshape(shape),
            max_vals=self._max_values().reshape(shape),
            scalar_manager=self.scalar_manager,
        )

    def reshape(self, *shape: Sequence[int]) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        output_values = self._values().reshape(*shape)
        target_shape = output_values.shape

        return InitialGammaTensor(
            values=output_values,
            entities=self._entities().reshape(target_shape),
            min_vals=self._min_values().reshape(target_shape),
            max_vals=self._max_values().reshape(target_shape),
            scalar_manager=self.scalar_manager,
        )

    def resize(self, *new_shape: Sequence[int]) -> IntermediateGammaTensor:

        """Currently, we are making all mutable operations (like resize) return a new tensor
        instead of modifying in place.
        This currently means resize == reshape."""
        return self.reshape(*new_shape)

    #     # relative
    #     from .initial_gamma import InitialGammaTensor
    #
    #     output_values = self._values()
    #     output_values.resize(new_shape, refcheck)
    #     shape = output_values.shape
    #     output_tensor = InitialGammaTensor(
    #         values=output_values,
    #         entities=self._entities().reshape(shape),
    #         max_vals=self._max_values().reshape(shape),
    #         min_vals=self._min_values().reshape(shape),
    #         scalar_manager=self.scalar_manager,
    #     )
    #
    #     # Copy all members from the new object
    #     self.__dict__ = output_tensor.__dict__
    # self.term_tensor = output_tensor.term_tensor
    # self.coeff_tensor = output_tensor.coeff_tensor
    # self.bias_tensor = output_tensor.bias_tensor

    def ravel(self, order: Optional[str] = "C") -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().ravel(order),
            entities=self._entities().ravel(order),
            min_vals=self._min_values().ravel(order),
            max_vals=self._max_values().ravel(order),
            scalar_manager=self.scalar_manager,
        )

    def squeeze(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().squeeze(axis),
            entities=self._entities().squeeze(axis),
            min_vals=self._min_values().squeeze(axis),
            max_vals=self._max_values().squeeze(axis),
            scalar_manager=self.scalar_manager,
        )

    def swapaxes(self, axis1: int, axis2: int) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().swapaxes(axis1, axis2),
            entities=self._entities().swapaxes(axis1, axis2),
            min_vals=self._min_values().swapaxes(axis1, axis2),
            max_vals=self._max_values().swapaxes(axis1, axis2),
            scalar_manager=self.scalar_manager,
        )

    # def partition(
    #     self,
    #     kth: Union[int, Tuple[int, ...]],
    #     axis: Optional[int] = -1,
    #     kind: Optional[str] = "introselect",
    #     order: Optional[Union[int, Tuple[int, ...]]] = None,
    # ) -> IntermediateGammaTensor:
    #     # relative
    #     from .initial_gamma import InitialGammaTensor
    #
    #     return InitialGammaTensor(
    #         values=self._values().partition(kth, axis, kind, order),
    #         entities=self._entities().partition(kth, axis, kind, order),
    #         min_vals=self._min_values().partition(kth, axis, kind, order),
    #         max_vals=self._max_values().partition(kth, axis, kind, order),
    #     )

    def compress(
        self,
        condition: List[bool],
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> PassthroughTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        if out:
            return InitialGammaTensor(
                values=self._values().compress(condition, axis, out),
                entities=self._entities().compress(condition, axis, out),
                min_vals=self._min_values().compress(condition, axis, out),
                max_vals=self._max_values().compress(condition, axis, out),
                scalar_manager=self.scalar_manager,  # TODO: Check if removing this creates a backdoor to data
            )
        else:
            # TODO: Check if "out" needs to be returned at all
            out = InitialGammaTensor(
                values=self._values().compress(condition, axis, out),
                entities=self._entities().compress(condition, axis, out),
                min_vals=self._min_values().compress(condition, axis, out),
                max_vals=self._max_values().compress(condition, axis, out),
                scalar_manager=self.scalar_manager,
            )
            return out

    def __and__(
        self, other: Union[np.ndarray, IntermediateGammaTensor]
    ) -> IntermediateGammaTensor:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=vals and other,
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
                    values=self_vals and other_vals,
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

    def __or__(
        self, other: Union[np.ndarray, IntermediateGammaTensor]
    ) -> IntermediateGammaTensor:
        if isinstance(other, np.ndarray):
            if is_broadcastable(self.shape, other.shape):
                # relative
                from .initial_gamma import InitialGammaTensor

                vals = self._values()
                tensor = InitialGammaTensor(
                    values=vals or other,
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
                    values=self_vals or other_vals,
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

    def take(
        self,
        indices: Optional[Union[int, Tuple[int, ...]]] = None,
        axis: Optional[int] = None,
        mode: str = "raise",
    ) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        if not axis:
            return InitialGammaTensor(
                values=self._values().take(indices, mode),
                entities=self._entities().take(indices, mode),
                min_vals=self._min_values().take(indices, mode),
                max_vals=self._max_values().take(indices, mode),
                scalar_manager=self.scalar_manager,
            )
        else:
            return InitialGammaTensor(
                values=self._values().take(indices, axis, mode),
                entities=self._entities().take(indices, axis, mode),
                min_vals=self._min_values().take(indices, axis, mode),
                max_vals=self._max_values().take(indices, axis, mode),
                scalar_manager=self.scalar_manager,
            )

    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> IntermediateGammaTensor:
        # Note: Currently NumPy returns a read only view, but plans to return a full copy in the future
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().diagonal(offset, axis1, axis2),
            entities=self._entities().diagonal(offset, axis1, axis2),
            min_vals=self._min_values().diagonal(offset, axis1, axis2),
            max_vals=self._max_values().diagonal(offset, axis1, axis2),
            scalar_manager=self.scalar_manager,
        )

    def put(
        self,
        indices: Sequence[int],  # Union[int, Tuple[int, ...], np.ndarray],
        values: Sequence[int],  # Union[int, Tuple[int, ...], np.ndarray],
        mode: Optional[str] = "raise",
    ) -> None:
        # relative
        from .initial_gamma import InitialGammaTensor

        new_values = self._values()

        # TODO: Check what happens with entities here, if data is replaced with public values?

        if isinstance(values, np.ndarray):
            for index, value in zip(indices, values):
                new_values[index] = value

            _ = InitialGammaTensor(
                values=new_values,
                entities=self._entities(),
                min_vals=self._min_values(),
                max_vals=self._max_values(),
                scalar_manager=self.scalar_manager,
            )

        else:
            raise NotImplementedError

    def trace(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().trace(offset, axis1, axis2),
            entities=self._entities().trace(offset, axis1, axis2),
            max_vals=self._max_values().trace(offset, axis1, axis2),
            min_vals=self._min_values().trace(offset, axis1, axis2),
            scalar_manager=self.scalar_manager,
        )

    def any(self) -> bool:
        for i in self._values():
            if i.any():
                return True
        return False

    def all(self) -> bool:
        for i in self._values():
            if not i.all():
                return False
        return True

    def __abs__(self) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=abs(self._values()),
            entities=self._entities(),
            max_vals=abs(self._max_values()),
            min_vals=abs(self._min_values()),
            scalar_manager=self.scalar_manager,
        )

    # def __divmod__(self, other: Union[int, np.ndarray]) -> IntermediateGammaTensor:
    #     # relative
    #     from .initial_gamma import InitialGammaTensor
    #
    #     return InitialGammaTensor(
    #         values=self._values() % other,
    #         entities=self._entities(),
    #         max_vals=self._max_values() % other,
    #         min_vals=self._min_values() % other,
    #     )

    def __floordiv__(self, other: Union[int, np.ndarray]) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values() // other,
            entities=self._entities(),
            max_vals=self._max_values() % other,
            min_vals=self._min_values() % other,
        )

    def cumsum(self, axis: Optional[int] = None) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().cumsum(axis),
            entities=self._entities().cumsum(axis),
            max_vals=self._max_values().cumsum(axis),
            min_vals=self._min_values().cumsum(axis),
            scalar_manager=self.scalar_manager,
        )

    def cumprod(self, axis: Optional[int] = None) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        return InitialGammaTensor(
            values=self._values().cumprod(axis),
            entities=self._entities().cumsum(
                axis
            ),  # entities get summed (combined), not multiplied
            max_vals=self._max_values().cumprod(axis),
            min_vals=self._min_values().cumprod(axis),
            scalar_manager=self.scalar_manager,
        )

    #
    # def __round__(self, n: Optional[int] = None) -> IntermediateGammaTensor:
    #     # relative
    #     from .initial_gamma import InitialGammaTensor
    #
    #     return InitialGammaTensor(
    #         values=self._values().__round__(n),
    #         entities=self._entities(),
    #         max_vals=self._max_values().__round__(n),
    #         min_vals=self._min_values().__round__(n),
    #     )

    def max(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        output_values = self._values().max(axis)
        indices = output_values.argmax(axis)

        # If indices returns a single value, then take simply returns an Entity or DSG instead of an array object
        # with an Entity/DSG inside. We will fix this by manually casting the results into arrays.

        output_entities = np.asarray(self._entities().take(indices))
        output_min = np.asarray(self._min_values().take(indices))
        output_max = np.asarray(self._max_values().take(indices))

        # TODO: Investigate if this can technically return a SEPT, if the max value only has 1 entity
        return InitialGammaTensor(
            values=output_values,
            entities=output_entities,
            max_vals=output_max,
            min_vals=output_min,
            scalar_manager=self.scalar_manager,
        )

    def min(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> IntermediateGammaTensor:
        # relative
        from .initial_gamma import InitialGammaTensor

        output_values = self._values().min(axis)
        indices = output_values.argmin(axis)

        # TODO: Investigate if this can technically return a SEPT, if the max value only has 1 entity
        return InitialGammaTensor(
            values=output_values,
            entities=np.asarray(self._entities().take(indices)),
            max_vals=np.asarray(self._max_values().take(indices)),
            min_vals=np.asarray(self._min_values().take(indices)),
            scalar_manager=self.scalar_manager,
        )
