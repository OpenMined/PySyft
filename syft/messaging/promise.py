"""This file contains the object we use to tell another worker that they will
receive an object with a certain ID. This object will also contain a set of
Plan objects which will be executed when the promise is kept (assuming
the plan has all of the inputs it requires). """

import syft as sy

class Promise:

    def __init__(self, id=None, obj_id=None, plans=set()):
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

        if id is None:
            id = sy.ID_PROVIDER.pop()

        self.id = id

        if obj_id is None:
            obj_id = sy.ID_PROVIDER.pop()

        self.obj_id = obj_id

        self.plans = plans