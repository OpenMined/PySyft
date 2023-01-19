# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import TYPE_CHECKING
from typing import Union

# third party
import jax.numpy as jnp
from numpy.typing import ArrayLike

# relative
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ..passthrough import AcceptableSimpleType
from ..passthrough import PassthroughTensor  # type: ignore
from .gamma_tensor_ops import GAMMA_TENSOR_OP

if TYPE_CHECKING:
    # relative
    from .gamma_tensor import GammaTensor

    GammaInputType = Union[GammaTensor, PassthroughTensor, AcceptableSimpleType]
else:
    GammaInputType = Union[PassthroughTensor, AcceptableSimpleType]


class SyftJaxOp:
    @property
    def func(self) -> Callable:
        raise NotImplementedError(f"{type(self)} func constructor not implemented yet")

    @staticmethod
    def can_chain(value: GammaInputType) -> bool:
        return hasattr(value, "reconstruct")


@serializable(recursive_serde=True)
class SyftTerminalNoop(SyftJaxOp):
    __attr_allowlist__ = ["phi_id"]

    def __init__(self, phi_id: UID) -> None:
        self.phi_id = phi_id

    @property
    def func(self) -> Callable:
        def reconstruct(state: Dict) -> ArrayLike:
            return state[self.phi_id]

        return reconstruct

    def __repr__(self) -> str:
        return f"{type(self).__name__}<state[self.phi_id]>"


@serializable(recursive_serde=True)
class SyftJaxInfixOp(SyftJaxOp):
    __attr_allowlist__ = ["jax_op", "left", "right"]

    def __init__(
        self,
        jax_op: GAMMA_TENSOR_OP,
        left: GammaInputType,
        right: GammaInputType,
    ) -> None:
        self.jax_op = jax_op
        self.left = left
        self.right = right

    @property
    def func(self) -> Callable:
        def infix_closure(state: Dict) -> GammaInputType:
            jax_func = getattr(jnp, self.jax_op.value)
            left = self.left.reconstruct(state) if self.chain_left else self.left
            right = self.right.reconstruct(state) if self.chain_right else self.right

            # TODO: remove this leaky abstraction
            # @Teo: how do we normally get past jax and our types "is not a valid JAX type."?
            if hasattr(left, "child"):
                left = left.child
            if hasattr(right, "child"):
                right = right.child
            return jax_func(left, right)

        return infix_closure

    @property
    def chain_left(self) -> bool:
        return self.can_chain(self.left)

    @property
    def chain_right(self) -> bool:
        return self.can_chain(self.right)

    def __repr__(self) -> str:
        left = "self.reconstruct(state)" if self.chain_left else "self"
        right = "other.reconstruct(state)" if self.chain_right else "other"
        return f"{type(self).__name__}<jnp.{self.jax_op}({left}, {right})>"


@serializable(recursive_serde=True)
class SyftJaxUnaryOp(SyftJaxOp):

    __attr_allowlist__ = ["jax_op", "operand", "args", "kwargs", "operand_before"]

    def __init__(
        self,
        jax_op: GAMMA_TENSOR_OP,
        operand: GammaInputType,
        args: Any,
        kwargs: Any,
        operand_before: bool = True,
    ) -> None:
        self.jax_op = jax_op
        self.operand = operand
        self.args = args
        self.kwargs = kwargs
        self.operand_before = operand_before

    @property
    def func(self) -> Callable:
        def unary_closure(state: Dict) -> GammaInputType:
            jax_func = getattr(jnp, self.jax_op.value)
            operand = (
                self.operand.reconstruct(state) if self.can_chain else self.operand  # type: ignore
            )
            if hasattr(operand, "child"):
                operand = operand.child
            if not self.operand_before:
                return jax_func(*self.args, operand, **self.kwargs)
            return jax_func(operand, *self.args, **self.kwargs)

        return unary_closure
