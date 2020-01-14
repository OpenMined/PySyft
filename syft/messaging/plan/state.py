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

    def __init__(self, owner, plan=None, state_placeholders=None):
        self.owner = owner
        self.plan = plan
        self.state_placeholders = state_placeholders or []

    def __str__(self):
        """Returns the string representation of the State."""
        out = "<"
        out += "State:"
        for state_placeholder in self.state_placeholders:
            out += " {}".format(state_placeholder)
        out += ">"
        return out

    def __repr__(self):
        return self.__str__()

    def tensors(self) -> List:
        """
        Fetch and return all the state elements.
        Perform a check of consistency on the elements ids.
        """
        tensors = []
        for placeholder in self.state_placeholders:
            tensor = placeholder.child
            tensors.append(tensor)
        return tensors

    def clone_state_dict(self) -> Dict:
        """
        Return a clone of the state elements. Tensor ids are kept.
        """
        return {placeholder.id: placeholder.child.clone() for placeholder in self.state_placeholders}

    def copy(self) -> "State":
        state = State(owner=self.owner, state_placeholders=self.state_placeholders.copy())
        return state

    def read(self):
        """
        Return state elements
        """
        tensors = self.tensors()
        if sy.hook.trace:
            for tensor in tensors:
                if tensor.id not in self.plan.placeholders.keys():
                    placeholder = sy.PlaceHolder(tags={f'#{self.plan.var_count + 1}'})
                    placeholder.instanciate(tensor)
                    self.plan.placeholders[tensor.id] = placeholder
                    self.state_placeholders.append(placeholder)
                    placeholder.tags.add('#state')
                    self.plan.var_count += 1
                else:
                    print('WARNING: tensor used before opened from state', tensor)
        return tensors

    def set_(self, state_dict):
        """
        Reset inplace the state by feeding it a dict of tensors or params
        """
        #assert list(self.state_placeholders) == list(state_dict.keys())

        for placeholder_id, new_tensor in state_dict.items():
            for placeholder in self.state_placeholders:
                if placeholder.id == placeholder_id:
                    placeholder.instanciate(new_tensor)

    @staticmethod
    def create_grad_if_missing(tensor):
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is None:
            o = tensor.sum()
            o.backward()
            if tensor.grad is not None:
                tensor.grad -= tensor.grad


    # def fix_precision_(self, *args, **kwargs):
    #     for tensor in self.tensors():
    #         self.create_grad_if_missing(tensor)
    #         tensor.fix_precision_(*args, **kwargs)
    #
    # def float_precision_(self):
    #     for tensor in self.tensors():
    #         tensor.float_precision_()
    #
    # def share_(self, *args, **kwargs):
    #     for tensor in self.tensors():
    #         self.create_grad_if_missing(tensor)
    #         tensor.share_(*args, **kwargs)

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
    def simplify(worker: AbstractWorker, state: "State") -> tuple:
        """
        Simplify the plan's state when sending a plan
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, state.state_placeholders),
            sy.serde.msgpack.serde._simplify(worker, state.tensors()),
        )

    @staticmethod
    def detail(worker: AbstractWorker, state_tuple: tuple) -> "State":
        """
        Reconstruct the plan's state from the state elements and supposed
        ids.
        """
        state_ids, state_elements = state_tuple
        state_ids = sy.serde.msgpack.serde._detail(worker, state_ids)
        state_elements = sy.serde.msgpack.serde._detail(worker, state_elements)

        for state_id, state_element in zip(state_ids, state_elements):
            worker.register_obj(state_element, obj_id=state_id)

        state = State(owner=worker, plan=None, state_placeholders=state_ids)
        return state
