# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np
from sympy.ntheory.factor_ import factorint

# relative
from ...adp.publish import publish
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.recursive import RecursiveSerde
from ...tensor.passthrough import PassthroughTensor  # type: ignore
from ...tensor.passthrough import is_acceptable_simple_type  # type: ignore


class IntermediateGammaTensor(PassthroughTensor, RecursiveSerde):

    __attr_allowlist__ = [
        "term_tensor",
        "coeff_tensor",
        "bias_tensor",
        "scalar_manager",
        "child",
    ]

    def __init__(
        self,
        term_tensor: np.ndarray,
        coeff_tensor: np.ndarray,
        bias_tensor: np.ndarray,
        scalar_manager: VirtualMachinePrivateScalarManager = VirtualMachinePrivateScalarManager(),
    ) -> None:
        super().__init__(term_tensor)
        self.term_tensor = term_tensor
        self.coeff_tensor = coeff_tensor
        self.bias_tensor = bias_tensor
        self.scalar_manager = scalar_manager

    @property
    def shape(self) -> Tuple[int]:
        return self.term_tensor.shape[:-1]

    @property
    def full_shape(self) -> Tuple[int]:
        return self.term_tensor.shape

    def publish(self, acc: Any, sigma: float, user_key: VerifyKey) -> np.ndarray:
        return np.array(
            publish(scalars=self.flat_scalars, acc=acc, sigma=sigma, user_key=user_key)
        ).reshape(self.shape)

    @property
    def flat_scalars(self) -> List[Any]:
        flattened_terms = self.term_tensor.reshape(-1, self.term_tensor.shape[-1])
        flattened_coeffs = self.coeff_tensor.reshape(-1, self.coeff_tensor.shape[-1])
        flattened_bias = self.bias_tensor.reshape(-1)

        scalars = list()

        for i in range(len(flattened_terms)):
            single_poly_terms = flattened_terms[i]
            single_poly_coeffs = flattened_coeffs[i]
            single_poly_bias = flattened_bias[i]

            scalar = single_poly_bias

            for j in range(len(single_poly_terms)):
                term = single_poly_terms[j]
                coeff = single_poly_coeffs[j]

                for prime, n_times in factorint(term).items():
                    input_scalar = self.scalar_manager.prime2symbol[prime]

                    scalar = scalar + (input_scalar * n_times * coeff)

            scalars.append(scalar)

        return scalars

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> IntermediateGammaTensor:

        new_term_tensor = np.swapaxes(self.term_tensor, axis, -1).squeeze(axis)
        new_coeff_tensor = np.swapaxes(self.coeff_tensor, axis, -1).squeeze(axis)
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
            term_tensor = np.concatenate([self.term_tensor, other.term_tensor], axis=-1)
            coeff_tensor = np.concatenate(
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
                    new_term = np.expand_dims(
                        self.term_tensor[..., self_dim]
                        * other.term_tensor[..., other_dim],
                        -1,
                    )
                    terms.append(new_term)

            for self_dim in range(self.term_tensor.shape[-1]):
                new_term = np.expand_dims(self.term_tensor[..., self_dim], -1)
                terms.append(new_term)

            for other_dim in range(self.term_tensor.shape[-1]):
                new_term = np.expand_dims(other.term_tensor[..., self_dim], -1)
                terms.append(new_term)

            term_tensor = np.concatenate(terms, axis=-1)

            coeffs = list()
            for self_dim in range(self.coeff_tensor.shape[-1]):
                for other_dim in range(other.coeff_tensor.shape[-1]):
                    new_coeff = np.expand_dims(
                        self.coeff_tensor[..., self_dim]
                        * other.coeff_tensor[..., other_dim],
                        -1,
                    )
                    coeffs.append(new_coeff)

            for self_dim in range(self.coeff_tensor.shape[-1]):
                new_coeff = np.expand_dims(
                    self.coeff_tensor[..., self_dim] * other.bias_tensor, -1
                )
                coeffs.append(new_coeff)

            for other_dim in range(self.coeff_tensor.shape[-1]):
                new_coeff = np.expand_dims(
                    other.coeff_tensor[..., self_dim] * self.bias_tensor, -1
                )
                coeffs.append(new_coeff)

            coeff_tensor = np.concatenate(coeffs, axis=-1)

            bias_tensor = self.bias_tensor * other.bias_tensor

        # TODO: Step 2: Reduce dimensionality if possible (look for duplicates)
        return IntermediateGammaTensor(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=self.scalar_manager,
        )
