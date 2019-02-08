import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor


class MultiPointerTensor(AbstractTensor):
    ""

    def __init__(
        self,
        parent: AbstractTensor = None,
        location=None,
        id_at_location=None,
        register=False,
        owner=None,
        id=None,
        garbage_collect_data=True,
        shape=None,
        point_to_attr=None,
        tags=None,
        description=None,
        children=None,
    ):

        super().__init__(tags, description)

        self.location = location
        self.id_at_location = id_at_location
        self.owner = owner
        self.id = id
        self.garbage_collect_data = garbage_collect_data
        self.point_to_attr = point_to_attr

        self.child = {}
        for c in children:
            assert c.shape == children[0].shape
            self.child[c.location.id] = c

    @property
    def shape(self):
        """This method returns the shape of the data being pointed to.
        This shape information SHOULD be cached on self._shape, but
        occasionally this information may not be present. If this is the
        case, then it requests the shape information from the remote object
        directly (which is inefficient and should be avoided)."""

        return list(self.child.values())[0].shape

    def get(self, sum_results=False):

        results = list()
        for v in self.child.values():
            results.append(v.get())

        if sum_results:
            return sum(results)

        return results
