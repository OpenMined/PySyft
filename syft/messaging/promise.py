"""This file contains the object we use to tell another worker that they will
receive an object with a certain ID. This object will also contain a set of
Plan objects which will be executed when the promise is kept (assuming
the plan has all of the inputs it requires). """

import syft as sy

from abc import ABC
from syft.workers.abstract import AbstractWorker


class Promise(ABC):
    def __init__(self, owner=None, id=None, obj_id=None, obj_type=None, plans=None):
        """Initialize a Promise with a unique ID and a set of (possibly empty) plans

        A Promise is a data-structure which indicates that "there will be an object
        with this ID at some point, and when you get it, use it to execute these plans".

        As such, the promise has an ID itself and it also has an ID of the tensor it's
        waiting for. Some promises self-destruct when they're kept, others can be kept
        multiple times. TODO: add support for promises which can be kept multiple times.

        However, it's important to know that some Plans are actually waiting on multiple
        objects before they can be executed. Thus, it's possible that you might call
        .keep() on a promise and nothing will happen because all of the plans are also
        waiting on other plans to be kept before they execute.

        Args:
            id (int): the id of the promise
            obj_id (int): the id of the object the promise is waiting for
            plans (set): a set of the plans waiting on the promise to be kept
        Example:
            future_x = Promise()
        """

        self.owner = owner

        if id is None:
            id = sy.ID_PROVIDER.pop()

        self._id = id

        if obj_id is None:
            obj_id = sy.ID_PROVIDER.pop()
        # TODO:
        # In which case would we need to pre-set an id for the object?
        # Can remove that if no use case
        # Else is it still ok to generate new id for promises kept several times?
        self.obj_id = obj_id
        self.queue_obj_ids = []

        self.obj_type = obj_type

        if plans is None:
            plans = set()

        self.plans = plans

    def keep(self, obj):
        print(f"keep {type(self)} {self.id}")
        if obj.type() != self.obj_type:
            raise TypeError(
                "keep() was called with an object of incorrect type (not the type that was promised)"
            )

        obj.id = self.obj_id

        if self.id in self.owner._objects:
            self.owner.register_obj(obj)

        self.queue_obj_ids.append(obj.id)
        # Generate new id for next time promise is kept
        self.obj_id = sy.ID_PROVIDER.pop()

        # If some plans were waiting for this promise...
        for plan in self.plans:
            # ... tell them that the promise has been kept.
            plan.args_promised[self.id].append(obj)  # TODO should I put obj_id in dict instead?

            # ... and execute them if it was the last argument they were waiting for.
            if plan.has_args_fulfilled():
                # Collect args
                args = [
                    plan.args_promised[arg_id].pop(0)
                    if arg_id in plan.args_promised
                    else self.owner._objects[arg_id]
                    for arg_id in plan.arg_ids
                ]
                """
                args = []
                for arg_id in plan.arg_ids:
                    if arg_id in plan.args_promised:
                        args.append(plan.args_promised[arg_id].pop(0))
                    else:
                        args.append(self.owner._objects[arg_id])
                """
                result = plan(*args)
                plan.output_promise.keep(result)

        return obj

    def value(self):
        """ Returns the next object in the queue of results.
        """
        if not self.queue_obj_ids:
            # If the promise has still not been kept
            # or if the queue of results has been emptied
            # TODO this doesn't work as I want with pointerTensors
            return None
        # TODO something like .pop() and/or .top()
        # return self.owner._objects[self.obj_id]
        ret_id = self.queue_obj_ids.pop(0)
        ret = self.owner._objects[ret_id]
        self.owner.rm_obj(ret_id)
        return ret

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id
        return new_id

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
