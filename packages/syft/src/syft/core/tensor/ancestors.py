# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
import uuid

# third party
from nacl.signing import VerifyKey
from numpy.typing import ArrayLike

# relative
# syft relative
from ..adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from .manager import TensorChainManager
from .passthrough import PassthroughTensor
from .passthrough import is_acceptable_simple_type

_SingleEntityPhiTensorRef = None


def _SingleEntityPhiTensor() -> Type[PassthroughTensor]:
    global _SingleEntityPhiTensorRef
    if _SingleEntityPhiTensorRef is None:
        # syft relative
        # relative
        from .autodp.single_entity_phi import SingleEntityPhiTensor

        _SingleEntityPhiTensorRef = SingleEntityPhiTensor
    return _SingleEntityPhiTensorRef


_RowEntityPhiTensorRef = None


def _RowEntityPhiTensor() -> Type[PassthroughTensor]:
    global _RowEntityPhiTensorRef
    if _RowEntityPhiTensorRef is None:
        # syft relative
        # relative
        from .autodp.row_entity_phi import RowEntityPhiTensor

        _RowEntityPhiTensorRef = RowEntityPhiTensor
    return _RowEntityPhiTensorRef


_AutogradTensorRef = None


def _AutogradTensor() -> Type[PassthroughTensor]:
    global _AutogradTensorRef
    if _AutogradTensorRef is None:
        # syft relative
        # relative
        from .autograd.tensor import AutogradTensor

        _AutogradTensorRef = AutogradTensor
    return _AutogradTensorRef


class AutogradTensorAncestor(TensorChainManager):
    """Inherited by any class which might have or like to have AutogradTensor in its chain
    of .child objects"""

    @property
    def grad(self):  # type: ignore
        child_gradient = self.child.grad
        if child_gradient is None:
            return None
        return self.__class__(child_gradient)

    @property
    def requires_grad(self) -> bool:
        return self.child.requires_grad

    def backward(self, grad=None):  # type: ignore

        AutogradTensor = _AutogradTensor()

        # TODO: @Madhava question, if autograd(requires_grad=True) is not set
        # we still end up in here from AutogradTensorAncestor but child.backward
        # has no backprop_id
        if isinstance(self.child, AutogradTensorAncestor) or isinstance(
            self.child, AutogradTensor
        ):

            if grad is not None and not is_acceptable_simple_type(grad):
                grad = grad.child

            return self.child.backward(grad, backprop_id=uuid.uuid4())  # type: ignore
        else:
            raise Exception(
                "No AutogradTensor found in chain, but backward() method called."
            )

    def autograd(self, requires_grad: bool = True) -> AutogradTensorAncestor:
        AutogradTensor = _AutogradTensor()

        self.push_abstraction_top(AutogradTensor, requires_grad=requires_grad)  # type: ignore

        return self


class PhiTensorAncestor(TensorChainManager):
    """Inherited by any class which might have or like to have SingleEntityPhiTensor in its chain
    of .child objects"""

    def __init__(self) -> None:
        pass

    @property
    def min_vals(self):  # type: ignore
        return self.__class__(self.child.min_vals)

    @property
    def max_vals(self):  # type: ignore
        return self.__class__(self.child.max_vals)

    @property
    def gamma(self):  # type: ignore
        return self.__class__(self.child.gamma)

    def publish(self, acc: Any, sigma: float, user_key: VerifyKey) -> PhiTensorAncestor:
        return self.__class__(
            self.child.publish(acc=acc, sigma=sigma, user_key=user_key)
        )

    def private(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        scalar_manager: VirtualMachinePrivateScalarManager = VirtualMachinePrivateScalarManager(),
        entities: Optional[List] = None,
        entity: Optional[Dict[str, Any]] = None,
    ) -> PhiTensorAncestor:
        """ """

        if entity is not None:
            # if there's only one entity - push a SingleEntityPhiTensor

            if isinstance(min_val, (float, int)):
                min_vals = (self.child * 0) + min_val
            else:
                raise Exception(
                    "min_val should be a float, got " + str(type(min_val)) + " instead."
                )

            if isinstance(max_val, (float, int)):
                max_vals = (self.child * 0) + max_val
            else:
                raise Exception(
                    "min_val should be a float, got " + str(type(min_val)) + " instead."
                )

            self.push_abstraction_top(
                _SingleEntityPhiTensor(),
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=scalar_manager,  # type: ignore
            )

        # if there's row-level entities - push a RowEntityPhiTensor
        elif entities is not None and len(entities) == self.shape[0]:

            class_type = _SingleEntityPhiTensor()

            new_list = list()
            for i, entity in enumerate(entities):

                if isinstance(min_val, (float, int)):
                    min_vals = (self.child[i : i + 1] * 0) + min_val  # noqa: E203
                else:
                    raise Exception(
                        "min_val should be a float, got "
                        + str(type(min_val))
                        + " instead."
                    )

                if isinstance(max_val, (float, int)):
                    max_vals = (self.child[i : i + 1] * 0) + max_val  # noqa: E203
                else:
                    raise Exception(
                        "min_val should be a float, got "
                        + str(type(min_val))
                        + " instead."
                    )

                value = self.child[i : i + 1]  # noqa: E203

                new_list.append(
                    class_type(
                        child=value,
                        entity=entity,
                        min_vals=min_vals,
                        max_vals=max_vals,
                        scalar_manager=scalar_manager,
                    )
                )

            self.replace_abstraction_top(_RowEntityPhiTensor(), rows=new_list)  # type: ignore

        # TODO: if there's element-level entities - push all elements with PhiScalars
        else:

            raise Exception(
                "If you're passing in mulitple entities, please pass in one entity per row."
            )

        return self
