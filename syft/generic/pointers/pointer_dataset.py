from typing import List
from typing import Union

import syft as sy
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.abstract import AbstractWorker


class PointerDataset(ObjectPointer):
    def __init__(
        self,
        location: "AbstractWorker" = None,
        id_at_location: Union[str, int] = None,
        owner: "AbstractWorker" = None,
        garbage_collect_data: bool = True,
        id: Union[str, int] = None,
        tags: List[str] = None,
        description: str = None,
    ):
        if owner is None:
            owner = sy.framework.hook.local_worker
        super().__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            garbage_collect_data=garbage_collect_data,
            id=id,
            tags=tags,
            description=description,
        )

    def get(self):
        dataset = self.location.get_obj(self.id_at_location)
        return dataset

    def data(self):
        command = ("get_data", self.id_at_location, [], {})
        ptr = self.owner.send_command(message=command, recipient=self.location)
        return ptr

    def targets(self):
        command = ("get_targets", self.id_at_location, [], {})
        ptr = self.owner.send_command(message=command, recipient=self.location)
        return ptr

    def wrap(self):
        return self
