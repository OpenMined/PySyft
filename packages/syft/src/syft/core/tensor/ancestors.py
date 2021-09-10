# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import uuid

# third party
from nacl.signing import VerifyKey
import numpy as np
from numpy.typing import ArrayLike

# relative
from ..adp.entity import Entity
from ..adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from .manager import TensorChainManager
from .passthrough import PassthroughTensor  # type: ignore
from .passthrough import is_acceptable_simple_type  # type: ignore

_SingleEntityPhiTensorRef = None

# stdlib
import textwrap


def _SingleEntityPhiTensor() -> Type[PassthroughTensor]:
    global _SingleEntityPhiTensorRef
    if _SingleEntityPhiTensorRef is None:
        # relative
        # relative
        from .autodp.single_entity_phi import SingleEntityPhiTensor

        _SingleEntityPhiTensorRef = SingleEntityPhiTensor
    return _SingleEntityPhiTensorRef


_RowEntityPhiTensorRef = None


def _RowEntityPhiTensor() -> Type[PassthroughTensor]:
    global _RowEntityPhiTensorRef
    if _RowEntityPhiTensorRef is None:
        # relative
        # relative
        from .autodp.row_entity_phi import RowEntityPhiTensor

        _RowEntityPhiTensorRef = RowEntityPhiTensor
    return _RowEntityPhiTensorRef


_AutogradTensorRef = None


def _AutogradTensor() -> Type[PassthroughTensor]:
    global _AutogradTensorRef
    if _AutogradTensorRef is None:
        # relative
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


def entity_creation_wizard(data) -> List[Any]:

    w = textwrap.TextWrapper(initial_indent="\t", subsequent_indent="\t")

    welcome_msg = "Welcome to the Data Subject Annotation Wizard!!!"

    description1 = """You've arrived here because you called Tensor.private() without passing in any entities!
Since the purpose of .private() is to add metadata for the support of automatic differential
privacy budgeting, you need to describe which parts of your Tensor correspond to which
real-world data subjects (entities) whose privacy you want to protect. This is the only
way the system knows, for example, that it costs twice as much privacy budget when twice
as much of your data (say, 2 rows instead of 1 row) refer to the same entity."""

    description2 = """Entities can be people (such as a medical patient), places (such as a family's address), or
even organizations (such as a business, state, or country). If you're not sure what kind of entity
to include, just ask yourself the question, "who am I trying to protect the privacy of?". If it's
an organization, make one entity per organization. If it's people, make one entity per person.
If it's a group of people who are somehow similar/linked to each other (such as a family),
make each entity a different group. For more information on differential privacy, see OpenMined's
course on the subject: https://courses.openmined.org/"""

    description3 = """Since you didn't pass in entities into .private() (or you did so incorrectly), this wizard is
going to guide you through the process of annotating your data with entities."""

    description4 = """In this wizard, we're going to ask you for *unique identifiers* which refer to the entities
in your data. While the unique identifiers need not be personal data (they can be random strings of letters and numbers
if you like). It is ESSENTIAL that you use the same identifier when referring to the same entity in the
data that you never accidentally refer to two entities by the same identifier. Additionally, if you plan
to do any kind of data JOIN with another dataset, it is ESSENTIAL that you are using the same unique
identifiers for entities as the data you're joining with. Since these unique identifiers may be personal
information, PySyft might not be able to detect if two tensors are using different identifiers for the
same person."""

    description5 = """So, in this tutorial we're going to be asking you to specify Unique Identifiers (UIDs) for each entity
in your data. This could be an email, street address, or any other string that identifies someone
uniquely in your data and in the data you intend to use with your data (if any)."""

    print("\t" + "=" * 69)
    print(w.fill(welcome_msg))
    print("\t" + "=" * 69)
    print()
    print(w.fill(description1))
    print()
    print(w.fill(description2))
    print()
    print(w.fill(description3))
    print()
    print(w.fill(description4))
    print()
    print(w.fill(description5))
    print()

    print("\tDo you understand, and are you ready to proceed? (yes/no)")
    consent = str(input("\t"))

    if consent == "no":
        raise Exception("User cancelled entity creation wizard!")

    print("\tExcellent! Let's begin!\n")
    # print("\tYou passed in a tensor with the shape:" + str(data.shape))
    print()

    print("\t" + "_" * 69)
    print()

    print(w.fill("Question 1: Is this entire tensor referring to the same entity?"))
    print()
    print(w.fill("Examples:"))
    print("\t - a single medical scan of one patient")
    print("\t - a single spreadsheet of proprietary statistics about a business")
    print("\t - a tensor of facts about a country")
    print()
    print(
        w.fill(
            """(if the tensor is about one entity, but it also contains multiple other entities within,
such as a tensor about all the customers of one business, ask yourself, are you trying to
protect the people or the business)"""
        )
    )
    print()
    print(
        w.fill(
            "If yes, write the UID of the entity this data is about, otherwise write 'no'."
        )
    )
    single_uid = input("\t")

    if single_uid != "no":
        print()
        print(
            w.fill(
                "Excellent! Your data will be annotated as referring to:"
                + str(single_uid)
            )
        )
        print()
        print(
            w.fill(
                "Congratulations! You're all done with the Data Subject Annotation Wizard!!!"
            )
        )
        print()
        print("\t" + "=" * 69)
        return [single_uid]

    print("\t" + "_" * 69)
    print()
    print(
        w.fill(
            "Question 2: Does each row correspond to an entity, perhaps with occasional repeats (yes/no)?"
        )
    )
    answer = str(input("\t"))

    print()

    print("\t" + "_" * 69)
    print(data)

    return ["asdf"]


class PhiTensorAncestor(TensorChainManager):
    """Inherited by any class which might have or like to have SingleEntityPhiTensor in its chain
    of .child objects"""

    def __init__(self, child: Any) -> None:
        self.child = child

    @property
    def shape(self) -> List[int]:
        return self.child.shape

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
            entities: Optional[Any] = None,
            skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:

        return self.copy()._private(min_val=min_val,
                                   max_val=max_val,
                                   scalar_manager=scalar_manager,
                                   entities=entities,
                                   skip_blocking_checks=skip_blocking_checks)

    def _private(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        scalar_manager: VirtualMachinePrivateScalarManager = VirtualMachinePrivateScalarManager(),
        entities: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        """ """

        # PHASE 1: RUN CHECKS

        # Check 1: Is self.child a compatible type? We only support DP and SMPC for a few types.
        if (
            not isinstance(self.child, np.ndarray)
            or getattr(self.child, "dtype", None) != np.int32
        ):

            msg = (
                "At present, you can only call .private() "
                + "on syft.Tensor objects wrapping np.int32 arrays. You called it on a "
                + "syft.Tensor wrapping a "
                + str(type(self.child))
            )

            if isinstance(self.child, np.ndarray):
                msg += " with dtype:" + str(getattr(self.child, "dtype", None))

            raise TypeError(msg)

        # Check 2: If entities == None, then run the entity creation tutorial
        if entities is None:

            if skip_blocking_checks:
                raise Exception(
                    "Error: 'entities' argument to .private() must not be None!"
                )
            print(
                "ALERT: You didn't pass in any entities. Launching entity wizard...\n"
            )
            entities = entity_creation_wizard(self.child)

        # Check 3: If entities is a string, make it a list with one entity in it
        if isinstance(entities, str):
            entities = [Entity(entities)]
        elif isinstance(entities, Entity):
            entities = [entities]

        # Check 4: If entities are a list, are the items strings or Entity objects.
        # If they're strings lets create Entity objects.
        _entities = list()
        for e in entities:
            if isinstance(e, str):
                _entities.append(Entity(e))
            else:
                _entities.append(e)

        entities = _entities

        # PHASE 2: CREATE CHILD
        if len(entities) == 1:
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
                entity=entities[0],
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
