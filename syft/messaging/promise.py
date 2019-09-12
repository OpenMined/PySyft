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

        self.obj_id = obj_id

        self.obj_type = obj_type

        if plans is None:
            plans = set()

        self.plans = plans

        # by default, a Promise has not been kept when it is created.
        self.is_kept = False

        self.owner.obj_id2promise_id[self.obj_id] = self.id

    def keep(self, obj):
        print(f"keep {type(self)} {self.id}")
        if obj.type() != self.obj_type:
            raise TypeError(
                "keep() was called with an object of incorrect type (not the type that was promised)"
            )

        obj.id = self.obj_id

        # not necessary since we're about to delete the Promise anyway
        # but when we have Promises which you can call multiple times
        # some functionality like this might be necessary.
        # self.is_kept = True

        for plan in self.plans:

            plan.args_fulfilled[self.obj_id] = obj

            if plan.has_args_fulfilled():
                args = list(map(lambda arg_id: plan.args_fulfilled[arg_id], plan.arg_ids))
                result = plan(*args)
                self.result_promise.parent().keep(result)

        if self.id in self.owner._objects:
            #self.owner.rm_obj(self.id)
            self.owner.register_obj(obj)

        return obj

    def value(self):
        """ Returns the object the promise points to.
        """
        #TODO is there a better way to access to the objects of a worker?
        return self.owner._objects[self.obj_id]

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
