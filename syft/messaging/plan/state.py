from typing import List
from typing import Tuple
from typing import Union
from typing import Dict

import torch

import syft as sy
from syft.workers.abstract import AbstractWorker


class State(object):
    """The State is a Plan attribute and is used to send tensors along functions.

    It references Plan tensor or parameters attributes using their name, and make
    sure they are provided to remote workers who are sent the Plan.
    """

    def __init__(self, owner, plan=None, state_ids=None):
        self.owner = owner
        self.plan = plan
        self.state_ids = state_ids or []

    def __repr__(self):
        return "State: " + ", ".join(self.state_ids)

    def tensors(self) -> List:
        """
        Fetch and return all the state elements.
        Perform a check of consistency on the elements ids.
        """
        tensors = []
        for state_id in self.state_ids:
            tensor = self.owner.get_obj(state_id)
            assert tensor.id == state_id
            tensors.append(tensor)
        return tensors

    def clone_state_dict(self) -> Dict:
        """
        Return a clone of the state elements. Tensor ids are kept.
        """
        return {tensor.id: tensor.clone() for tensor in self.tensors()}

    def copy(self) -> "State":
        state = State(owner=self.owner, state_ids=self.state_ids.copy())
        return state

    def read(self):
        """
        Return state elements
        """
        tensors = []
        for state_id in self.state_ids:
            tensor = self.owner.get_obj(state_id)
            tensors.append(tensor)
        return tensors

    def set_(self, state_dict):
        """
        Reset inplace the state by feeding it a dict of tensors or params
        """
        assert list(self.state_ids) == list(state_dict.keys())

        for state_id, new_tensor in state_dict.items():
            tensor = self.owner.get_obj(state_id)

            with torch.no_grad():
                tensor.set_(new_tensor)

            tensor.child = new_tensor.child if new_tensor.is_wrapper else None
            tensor.is_wrapper = new_tensor.is_wrapper
            if tensor.child is None:
                delattr(tensor, "child")

    @staticmethod
    def create_grad_if_missing(tensor):
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is None:
            o = tensor.sum()
            o.backward()
            if tensor.grad is not None:
                tensor.grad -= tensor.grad

    def send_for_build(self, location, **kwargs):
        """
        Send functionality that can only be used when sending the state for
        building the plan. Other than this, you shouldn't need to send the
        state separately.
        """
        assert location.id == self.plan.id  # ensure this is a send for the build

        for tensor in self.tensors():
            self.create_grad_if_missing(tensor)
            tensor.send_(location, **kwargs)

    def fix_precision_(self, *args, **kwargs):
        for tensor in self.tensors():
            self.create_grad_if_missing(tensor)
            tensor.fix_precision_(*args, **kwargs)

    def float_precision_(self):
        for tensor in self.tensors():
            tensor.float_precision_()

    def share_(self, *args, **kwargs):
        for tensor in self.tensors():
            self.create_grad_if_missing(tensor)
            tensor.share_(*args, **kwargs)

    def get_(self):
        """
        Get functionality that can only be used when getting back state
        elements converted to additive shared tensors. Other than this,
        you shouldn't need to the get the state separately.
        """
        # TODO Make it only valid for AST
        for tensor in self.tensors():
            tensor.get_()

    @staticmethod
    def simplify(state: "State") -> tuple:
        """
        Simplify the plan's state when sending a plan
        """
        return (sy.serde._simplify(state.state_ids), sy.serde._simplify(state.tensors()))

    @staticmethod
    def detail(worker: AbstractWorker, state_tuple: tuple) -> "State":
        """
        Reconstruct the plan's state from the state elements and supposed
        ids.
        """
        state_ids, state_elements = state_tuple
        state_ids = sy.serde._detail(worker, state_ids)
        state_elements = sy.serde._detail(worker, state_elements)

        for state_id, state_element in zip(state_ids, state_elements):
            worker.register_obj(state_element, obj_id=state_id)

        state = State(owner=worker, plan=None, state_ids=state_ids)
        return state
