# from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Union

import syft as sy
from syft.serde.syft_serializable import SyftSerializable
from syft.generic.abstract.object import AbstractObject

# from syft.generic.abstract.pointer import AbstractPointer
from syft.workers.abstract import AbstractWorker

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.generic.abstract.pointer import AbstractPointer


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def send(
        self, destination: Union[AbstractWorker, str], garbage_collect_data=None
    ) -> "AbstractPointer":
        ptr_id = sy.ID_PROVIDER.pop()

        destination = self.owner.get_worker(destination)

        pointer = self.create_pointer(
            location=destination,
            id_at_location=self.id,
            owner=self.owner,
            ptr_id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        self.owner.send_obj(self, destination)

        return pointer

    # TODO: Make the `create_pointer` method below an abstract method. We can't do this yet, because
    # we're relying on the wrapper tensors to provide the functionality of creating pointers, which
    # means that many of the custom tensor types don't yet have pointer types or ways to create
    # pointers.

    # @abstractmethod
    def create_pointer(
        self,
        location: AbstractWorker = None,
        id_at_location: (str or int) = None,
        owner: AbstractWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        **kwargs,
    ) -> "AbstractPointer":
        pass
