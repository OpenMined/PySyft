import random

import torch

from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.workers.base import BaseWorker
from syft.codes import MSGTYPE
import syft as sy

from typing import List
from typing import Union


def make_plan(plan_blueprint):
    """For folks who would prefer to not use a decorator, they can use this function"""
    return func2plan(plan_blueprint)


def func2plan(plan_blueprint):
    """
    the @func2plan decorator - converts a function of pytorch code into a plan object
    which can be sent to any arbitrary worker.
    """

    plan = Plan(
        hook=sy.local_worker.hook,
        owner=sy.local_worker,
        id=random.randint(0, 1e10),
        name=plan_blueprint.__name__,
    )

    plan.plan_blueprint = plan_blueprint

    return plan


def method2plan(plan_blueprint):
    """
    the @method2plan decorator - converts a method containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.
    """

    plan = Plan(
        hook=sy.local_worker.hook,
        owner=sy.local_worker,
        id=random.randint(0, 1e10),
        name=plan_blueprint.__name__,
    )

    plan.plan_blueprint = plan_blueprint

    @property
    def method(self: object) -> Plan:
        """
        This property is a way to catch the self of the method and give it to the plan,
        it will be provided in the future calls as this is not automatic (the structure
        of @func2plan would not keep the self during the call)

        Args:
            self (object): an instance of a class

        Returns:
            the plan which is also a callable.

        Example:
            When you have your plan and that you do
            > plan(*args)
            First the property is call with the part "plan" and self is caught, plan is
            returned
            Then plan is called with "(*args)" and in the __call__ function of plan the
            self parameter is re-inserted
        """
        plan.self = self

        return plan

    return method


class Plan(BaseWorker):
    """This worker does not send messages or execute any commands. Instead,
    it simply records messages that are sent to it such that message batches
    (called 'Plans') can be created and sent once."""

    def __init__(self, hook, owner, name="", *args, **kwargs):
        super().__init__(hook=hook, *args, **kwargs)
        # Plan instance info
        self.name = name
        self.owner = owner
        # Info about the plan stored
        self.plan = list()
        self.readable_plan = list()
        self.arg_ids = list()
        self.result_ids = list()
        self.owner_when_built = None
        # Pointing info towards a remote plan
        self.location = None
        self.ptr_plan = None

        # Planworkers are registered by other worker, they must have information
        # to be retrieved by search functions
        self.tags = None
        self.description = None
        # For methods
        self.self = None

    def _send_msg(self, message, location):
        return location._recv_msg(message)

    def _recv_msg(self, bin_message):
        """
        Upon reception, the PlanWorker store in the plan all commands which can be
        executed lazily
        :param bin_message: the message of a command received
        :return: the None message serialized to specify the command was received
        """
        (some_type, (msg_type, contents)) = sy.serde.deserialize(bin_message, detail=False)

        if msg_type != MSGTYPE.OBJ:
            self.plan.append(bin_message)
            self.readable_plan.append((some_type, (msg_type, contents)))

        # we can't receive the results of a plan without
        # executing it. So, execute the plan.
        if msg_type in (MSGTYPE.OBJ_REQ, MSGTYPE.IS_NONE, MSGTYPE.GET_SHAPE):
            return self.__call__()

        return sy.serde.serialize(None)

    def build_plan(self, args):
        """
        The plan must be built with some input data, here args. When they
        are provided, they are sent to the plan worker, which executes its
        blueprint: each command of the blueprint is catched by _recv_msg
        and is used to fill the plan
        :param args: the input data
        """
        # The ids of args of the first call, which should be updated when
        # the function is called with new args
        self.arg_ids = list()
        local_args = list()
        for i, arg in enumerate(args):
            # Send only tensors (in particular don't send the "self" for methods)
            if isinstance(arg, torch.Tensor):
                self.owner.register_obj(arg)
                arg = arg.send(self)
                arg.child.garbage_collect_data = False
                self.arg_ids.append(arg.id_at_location)
            local_args.append(arg)

        res_ptr = self.plan_blueprint(*local_args)
        res_ptr.child.garbage_collect_data = False

        # The id where the result should be stored
        self.result_ids = [res_ptr.id_at_location]

        # Store owner that built the plan
        self.owner_when_built = self.owner

    def replace_ids(self, from_ids, to_ids):
        """
        Replace pairs of tensor ids in the plan stored
        :param from_ids: the left part of the pair: ids to change
        :param to_ids: the right part of the pair: ids to replace with
        """
        # for every pair of id
        for i in range(len(from_ids)):
            # for every message of the plan
            for j, msg in enumerate(self.readable_plan):
                # look for the old id and replace it with the new one
                self.readable_plan[j] = Plan._replace_message_ids(
                    obj=msg,
                    change_id=from_ids[i],
                    to_id=to_ids[i],
                    from_worker=self.id,
                    to_worker=self.owner.id,
                )
        return self

    def replace_worker_ids(self, from_worker_id, to_worker_id):
        """
        Replace occurrences of from_worker_id by to_worker_id in the plan stored
        Works also if those ids are encoded in bytes (for string)
        """
        for from_id, to_id in [
            (from_worker_id, to_worker_id),
            (from_worker_id.encode(), to_worker_id.encode()),
        ]:
            self.readable_plan = Plan._replace_message_ids(
                obj=self.readable_plan, change_id=-1, to_id=-1, from_worker=from_id, to_worker=to_id
            )

    @staticmethod
    def _replace_message_ids(obj, change_id, to_id, from_worker, to_worker):
        _obj = list()

        for i, item in enumerate(obj):
            if isinstance(item, int) and (item == change_id):
                _obj.append(to_id)

            elif isinstance(item, type(from_worker)) and (item == from_worker):
                _obj.append(to_worker)

            elif isinstance(item, (list, tuple)):
                _obj.append(
                    Plan._replace_message_ids(
                        obj=item,
                        change_id=change_id,
                        to_id=to_id,
                        from_worker=from_worker,
                        to_worker=to_worker,
                    )
                )

            else:
                _obj.append(item)

        return _obj

    def __call__(self, *args, **kwargs):
        """
        Call a plan execution with some arguments, and specify the ids where the result
        should be stored
        :return: The pointer to the result of the execution if the plan was already sent,
        else the None message serialized.
        """
        assert len(kwargs) == 0, "kwargs not supported for plan"
        result_ids = [random.randint(0, 1e10)]
        # Support for method hooked in plans
        if self.self is not None:
            args = [self.self] + list(args)
        return self.execute_plan(args, result_ids)

    def _update_args(self, args, result_ids):
        """Replace args and result_ids with the ones given.
        Updates the arguments ids and result ids used to execute
        the plan.
        Args:
            args: List of tensors.
            result_ids: Ids where the plan output will be stored.
        """
        arg_ids = [arg.id for arg in args]
        self.replace_ids(self.arg_ids, arg_ids)
        self.arg_ids = arg_ids

        self.replace_ids(self.result_ids, result_ids)
        self.result_ids = result_ids

    def _execute_plan(self):
        for message in self.readable_plan:
            bin_message = sy.serde.serialize(message, simplified=True)
            x = self.owner.recv_msg(bin_message)

    def _get_plan_output(self, result_ids, return_ptr=False):
        responses = []
        for return_id in result_ids:
            response = sy.PointerTensor(
                location=self.owner,
                id_at_location=return_id,
                owner=self,
                id=int(10e10 * random.random()),
            )
            responses.append(response if return_ptr else response.get())

        if len(responses) == 1:
            return responses[0]

        return responses

    def _execute_plan_locally(self, result_ids, *args, **kwargs):
        if self.owner != self.owner_when_built:
            self.build_plan(args)
        self._update_args(args, result_ids)
        self._execute_plan()
        responses = self._get_plan_output(result_ids)
        return responses

    def execute_plan(self, args, result_ids):
        """
        Control local or remote plan execution.
        If the plan doesn't have the plan built, first build it using the blueprint.
        Then if it has a remote location, send the plan to the remote location only the
        first time, request a remote plan execution with specific pointers and ids for
        storing the result, and return a pointer to the result of the execution.
        If the plan is local: update the plan with the result_ids and args ids given,
        run the plan and return the None message serialized.
        """
        # We build the plan only if needed
        first_run = self.readable_plan == []
        if first_run:
            self.build_plan(args)

        # If there is a location we need to send the plan to
        # we request the location to execute the plan
        if self.location:
            if self.ptr_plan is None:
                self.ptr_plan = self._send(self.location)
            response = self.request_execute_plan(result_ids, *args)
            return response

        # If the plan is local, we execute the plan and return the response
        if not self.location and self.owner == sy.hook.local_worker:
            return self._execute_plan_locally(result_ids, *args)

        # if the plan is not to be sent but is not local (ie owned by the local worker)
        # then it has been request to execute, so we update the plan with the correct
        # input and output ids and we run it
        elif not self.location and self.owner != sy.hook.local_worker:
            self._update_args(args, result_ids)
            self._execute_plan()

        return sy.serde.serialize(None)

    def request_execute_plan(self, response_ids, *args, **kwargs):
        """
        Send a request to execute the plan on the remote location
        :param response_ids: where the plan result should be stored remotely
        :param args: the arguments use as input data for the plan
        :return:
        """
        args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        args = [args, response_ids]
        command = ("execute_plan", self.ptr_plan, args, kwargs)

        response = self.owner.send_command(
            message=command, recipient=self.location, return_ids=response_ids
        )
        return response

    def send(self, location):
        """
        Mock send function that only specify that the Plan will have to be sent to location.
        In a way, when one calls .send(), this doesn't trigger a call to a remote worker, but
        just stores "a promise" that it will be sent (with _send()) later when the plan in
        called (and built)
        """
        if self.location is not None:
            raise NotImplementedError(
                "Can't send a Plan which already has a location, use .get() before"
            )
        else:
            self.location = location
        return self

    def _send(self, location):
        """
        Real send function that sends the Plan instance with its plan to location, only
        when the plan is built and that an execution is called, namely when it is necessary
        to send it
        """
        self.replace_worker_ids(self.owner.id, self.location.id)
        return self.owner.send(obj=self, workers=location)

    def get(self):
        """
        Mock get function: no call to remote worker is made, we just erase the information
        linking this plan to that remote worker.
        """
        self.replace_worker_ids(self.location.id, self.owner.id)
        self.location = None
        self.ptr_plan = None
        return self

    def describe(self, description):
        self.description = description
        return self

    def tag(self, *_tags):
        if self.tags is None:
            self.tags = set()

        for new_tag in _tags:
            self.tags.add(new_tag)
        return self

    def __str__(self):
        """Returns the string representation of PlanWorker.
        Note:
            __repr__ calls this method by default.
        """
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " " + str(self.name)
        out += " id:" + str(self.id)
        out += " owner:" + str(self.owner.id)

        if self.location:
            out += " location:" + str(self.location.id)

        if self.tags is not None and len(self.tags):
            out += " Tags:"
            for tag in self.tags:
                out += " " + str(tag)

        if len(self.readable_plan) > 0:
            out += " built"

        out += ">"

        return out
