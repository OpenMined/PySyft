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

    @property
    def data(self):
        command = ("get_data", self.id_at_location, [], {})
        ptr = self.owner.send_command(message=command, recipient=self.location).wrap()
        return ptr

    @property
    def targets(self):
        command = ("get_targets", self.id_at_location, [], {})
        ptr = self.owner.send_command(message=command, recipient=self.location).wrap()
        return ptr

    def wrap(self):
        return self

    def __repr__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name} | " f"owner: {str(self.owner.id)}, id:{self.id}"

        if self.point_to_attr is not None:
            out += "::" + str(self.point_to_attr).replace(".", "::")

        big_str = False

        if self.tags is not None and len(self.tags):
            big_str = True
            out += "\n\tTags: "
            for tag in self.tags:
                out += str(tag) + " "

        if big_str and hasattr(self, "shape"):
            out += "\n\tShape: " + str(self.shape)

        if self.description is not None:
            big_str = True
            out += "\n\tDescription: " + str(self.description).split("\n")[0] + "..."

        return out

    def __len__(self):
        command = ("__len__", self.id_at_location, [], {})
        len = self.owner.send_command(message=command, recipient=self.location)
        return len

    def __getitem__(self, index):
        command = ("__getitem__", self.id_at_location, [index], {})
        data_elem, target_elem = self.owner.send_command(message=command, recipient=self.location)
        return data_elem.wrap(), target_elem.wrap()
