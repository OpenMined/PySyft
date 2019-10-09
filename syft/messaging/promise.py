"""This file contains the object we use to tell another worker that they will
receive an object with a certain ID. This object will also contain a set of
Plan objects which will be executed when the promise is kept (assuming
the plan has all of the inputs it requires). """

import syft as sy

from abc import ABC
from syft.workers.abstract import AbstractWorker


class Promise(ABC):
    def __init__(self, owner=None, obj_id=None, obj_type=None, plans=None):
        """Initialize a Promise with a unique ID and a set of (possibly empty) plans

        A Promise is a data-structure which indicates that "there will be an object
        with this ID at some point, and when you get it, use it to execute these plans".

        As such, the promise has an ID itself and it also has an queue of object ids
        with which the promise has been kept.

        However, it's important to know that some Plans are actually waiting on multiple
        objects before they can be executed. Thus, it's possible that you might call
        .keep() on a promise and nothing will happen because all of the plans are also
        waiting on other promises to be kept before they execute.

        Args:
            id (int): the id of the promise
            plans (set): a set of the plans waiting on the promise to be kept
        Example:
            future_x = Promise()
        """

        self.owner = owner

        self.obj_type = obj_type
        self.queue_obj_ids = []

        if plans is None:
            plans = set()
        self.plans = plans

    def keep(self, obj):
        """ This method is used to keep a promise.
        This will register the object on the worker, add its id to the queue of the promise,
        and every plan waiting for this promise will try to execute if it can.
        """
        if obj.type() != self.obj_type:
            raise TypeError(
                "keep() was called with an object of incorrect type (not the type that was promised)"
            )

        if self.id in self.owner._objects:
            self.owner.register_obj(obj)

        self.queue_obj_ids.append(obj.id)

        # If some plans were waiting for this promise...
        for plan_id in self.plans:
            # If the promise has been moved, the plan waiting for it might not be on the same worker
            # TODO is this ok?
            if plan_id in self.owner._objects:
                plan = self.owner.get_obj(plan_id)
            else:
                continue

            # ... execute them if it was the last argument they were waiting for.
            if plan.has_args_fulfilled():
                # Collect args
                orig_ids = plan.procedure.arg_ids
                args = []
                ids_to_rm = []
                for i, arg_id in enumerate(plan.procedure.arg_ids):
                    if isinstance(self.owner._objects[arg_id].child, Promise):
                        id_to_add = self.owner.get_obj(arg_id).child.queue_obj_ids.pop(0)
                        ids_to_rm.append(id_to_add)
                    else:
                        id_to_add = arg_id
                    args.append(self.owner.get_obj(id_to_add))
                    # FIXME Ugly fix because I had id_to_add != self.owner.get_obj(id_to_add).id...
                    args[-1].id = id_to_add
                result = plan(*args)

                # ids of promises are changed automatically otherwise
                plan.procedure.update_ids(plan.procedure.arg_ids, orig_ids)
                plan.procedure.arg_ids = orig_ids

                # Remove objects from queues:
                for to_rm in ids_to_rm:
                    self.owner.rm_obj(to_rm)

                self.owner.get_obj(plan.promise_out_id).keep(result)

        return obj

    def is_kept(self):
        """ Check if promise has objects waiting to be used.
        This returns False if the queue of objects is empty.
        """
        return self.queue_obj_ids != []

    def value(self):
        """ Returns the next object in the queue of results.
        """
        if not self.queue_obj_ids:
            # If the promise has still not been kept
            # or if the queue of results has been emptied
            # TODO this doesn't work as I want with pointerTensors
            return None
        ret_id = self.queue_obj_ids.pop(0)
        ret = self.owner.get_obj(ret_id)
        self.owner.rm_obj(ret_id)
        return ret

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"<Promise({self.id}) promises Obj(id:{self.obj_id}) blocking {len(self.plans)} plans>"
        )

    @staticmethod
    def simplify(promise: "Promise") -> tuple:
        """
        This function takes the attributes of a Promise and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            promise (Promise): a Promise object
        Returns:
            tuple: a tuple holding the unique attributes of the promise
        Examples:
            data = simplify(promise)
        """
        return (promise.id, promise.obj_id, sy.serde._simplify(promise.plans))

    @staticmethod
    def detail(worker: AbstractWorker, promise_tuple: tuple) -> "Promise":
        """
        This function takes the simplified tuple version of this promise and converts
        it into a Promise. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            promise_tuple (Tuple): the raw information being detailed into a Promise
        Returns:
            promise (Promise): a Promise object.
        Examples:
            message = detail(sy.local_worker, promise_tuple)
        """
        # TODO: probably need to register the Promise
        return Promise(promise_tuple[0], promise_tuple[1], set(sy.serde._detail(promise_tuple[3])))
