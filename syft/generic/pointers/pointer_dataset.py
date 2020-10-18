from typing import List
from typing import Union

import syft as sy
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.abstract import AbstractWorker
from syft_proto.generic.pointers.v1.pointer_dataset_pb2 import PointerDataset as PointerDatasetPB


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
        self.federated = False  # flag whether it in a federated_dataset object
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
        ptr = self.owner.send_command(
            cmd_name="get_data", target=self.id_at_location, recipient=self.location
        ).wrap()
        return ptr

    @property
    def targets(self):
        ptr = self.owner.send_command(
            cmd_name="get_targets", target=self.id_at_location, recipient=self.location
        ).wrap()
        return ptr

    def wrap(self):
        return self

    def get(self, user=None, reason: str = "", deregister_ptr: bool = True):
        if self.federated:
            raise ValueError("use .get_dataset(worker) to get this dataset")
        dataset = super().get(user, reason, deregister_ptr)
        return dataset

    def __repr__(self):
        type_name = type(self).__name__
        out = (
            f"["
            f"{type_name} | "
            f"{str(self.owner.id)}:{self.id}"
            " -> "
            f"{str(self.location.id)}:{self.id_at_location}"
            f"]"
        )

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
        len = self.owner.send_command(
            cmd_name="__len__", target=self.id_at_location, recipient=self.location
        )
        return len

    def __getitem__(self, index):
        args = [index]
        data_elem, target_elem = self.owner.send_command(
            cmd_name="__getitem__",
            target=self.id_at_location,
            args_=tuple(args),
            recipient=self.location,
        )
        return data_elem.wrap(), target_elem.wrap()

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "PointerDataset") -> tuple:

        return (
            sy.serde.msgpack.serde._simplify(worker, ptr.id),
            sy.serde.msgpack.serde._simplify(worker, ptr.id_at_location),
            sy.serde.msgpack.serde._simplify(worker, ptr.location.id),
            sy.serde.msgpack.serde._simplify(worker, ptr.tags),
            sy.serde.msgpack.serde._simplify(worker, ptr.description),
            ptr.garbage_collect_data,
        )

    @staticmethod
    def detail(worker: AbstractWorker, ptr_tuble: tuple) -> "PointerDataset":
        obj_id, id_at_location, worker_id, tags, description, garbage_collect_data = ptr_tuble

        obj_id = sy.serde.msgpack.serde._detail(worker, obj_id)
        id_at_location = sy.serde.msgpack.serde._detail(worker, id_at_location)
        worker_id = sy.serde.msgpack.serde._detail(worker, worker_id)
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)

        # If the pointer received is pointing at the current worker, we load the dataset instead
        if worker_id == worker.id:
            dataset = worker.get_obj(id_at_location)

            return dataset
        # Else we keep the same Pointer
        else:
            location = sy.hook.local_worker.get_worker(worker_id)

            ptr = PointerDataset(
                location=location,
                id_at_location=id_at_location,
                owner=worker,
                tags=tags,
                description=description,
                garbage_collect_data=garbage_collect_data,
                id=obj_id,
            )

            return ptr

    @staticmethod
    def bufferize(worker, pointer_obj):
        """
        This method serializes a PointerDataset into a PointerDatasetPB.

        Args:
            pointer_obj (PointerDataset): input PointerDataset to be serialized.

        Returns:
            protobuf_script (PointerDatasetPB): serialized PointerDataset.
        """
        proto_pointer = PointerDatasetPB()
        sy.serde.protobuf.proto.set_protobuf_id(proto_pointer.object_id, pointer_obj.id)
        sy.serde.protobuf.proto.set_protobuf_id(proto_pointer.location_id, pointer_obj.location.id)
        sy.serde.protobuf.proto.set_protobuf_id(
            proto_pointer.object_id_at_location, pointer_obj.id_at_location
        )
        for tag in pointer_obj.tags:
            proto_pointer.tags.append(tag)

        if pointer_obj.description:
            proto_pointer.description = pointer_obj.description
        proto_pointer.garbage_collect_data = pointer_obj.garbage_collect_data
        return proto_pointer

    @staticmethod
    def unbufferize(worker, proto_pointer_obj: PointerDatasetPB):
        """
        This method deserializes PointerDatasetPB into a PointerDataset.

        Args:
            protobuf_script (PointerDatasetPB): input serialized PointerDatasetPB.

        Returns:
            loaded_module (PointerDataset): deserialized PointerDatasetPB.
        """
        obj_id = sy.serde.protobuf.proto.get_protobuf_id(proto_pointer_obj.object_id)
        id_at_location = sy.serde.protobuf.proto.get_protobuf_id(
            proto_pointer_obj.object_id_at_location
        )
        worker_id = sy.serde.protobuf.proto.get_protobuf_id(proto_pointer_obj.location_id)
        tags = proto_pointer_obj.tags

        description = proto_pointer_obj.description if proto_pointer_obj.description else None

        garbage_collect_data = proto_pointer_obj.garbage_collect_data

        if worker_id == worker.id:
            dataset = worker.get_obj(id_at_location)
            return dataset

        location = sy.hook.local_worker.get_worker(worker_id)

        return PointerDataset(
            location=location,
            id_at_location=id_at_location,
            owner=worker,
            tags=set(tags),
            description=description,
            garbage_collect_data=garbage_collect_data,
            id=obj_id,
        )

    @staticmethod
    def get_protobuf_schema():
        """
        This method returns the protobuf schema used for PointerDataset.

        Returns:
           Protobuf schema for PointerDataset.
        """
        return PointerDatasetPB
