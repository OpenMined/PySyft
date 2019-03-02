import syft as sy

import copy
import random


def make_plan(worker):
    """
    hook args and response for methods that hold the @make_plan decorator
    """
    # retrieve or create the local plan
    if hasattr(sy.hook.local_worker, "plan"):
        local_plan = sy.hook.local_worker.plan
    else:
        local_plan = sy.workers.Plan(sy.hook.local_worker, sy.hook, id="local_plan")

    # retrieve or create the remote plan
    if hasattr(worker, "plan"):
        remote_plan = worker.plan
    else:
        remote_plan = sy.workers.Plan(worker, sy.hook, id=f"{worker.id}_plan")

    # define the function which creates the plan
    def create_plan(plan_function):
        """
        Take as an input a function to be converted into a plan, makes it a plan
        using local_plan and send the plan each time the function is called to
        the remote worker via its remote plan
        :param plan_function:
        :return: a pointer to the function result
        """
        local_plan.readable_plan = []
        local_plan.plan = []

        # The ids of args of the first call, which should be updated when
        # the function is called with new args
        arg_ids = list()
        # The id where the result should be stored
        res_id_at_location = list()  # make a list to keep reference

        def execute_plan(*args):
            """
            At each call of the plan_function, first build the associated plan if it
            does not exist, then send the plan to the remote_plan worker, which will
            update the plan with the ids of the current args, executes it and put the
            result at the specified id location
            :param args: args to execute the plan_function on
            :return: a pointer to the result of plan_function(*args)
            """
            # Create the local plan if it not defined yet (so at the first call)
            if local_plan.plan == []:
                local_args = []
                # the plan need to be created on the local_plan worker so send all
                # args to him
                for i in range(len(args)):
                    local_arg = args[i].send(local_plan)
                    local_arg.child.garbage_collect_data = False
                    local_args.append(local_arg)
                    arg_ids.append(local_arg.id_at_location)
                # then run the plan_function with these args to build the plan
                res_ptr = plan_function(*local_args)
                res_ptr.child.garbage_collect_data = False
                res_id_at_location.append(res_ptr.id_at_location)

            # Send the local plan to the remote_plan worker (TODO: here just a copy)
            remote_plan.readable_plan = copy.deepcopy(local_plan.readable_plan)
            remote_plan.plan = copy.deepcopy(local_plan.plan)

            # Update plan with current args for every argument (TODO: remotely)
            for i in range(len(args)):
                from_id, to_id = arg_ids[i], args[i].id_at_location
                remote_plan.readable_plan = replace_ids(remote_plan.readable_plan, from_id, to_id)

            # Execute the plan
            remote_plan.execute_plan()

            # Return a pointer to the address where the plan result should be stored ont he remote worker
            return sy.PointerTensor(
                id=int(10e10 * random.random()),
                id_at_location=res_id_at_location[0],
                location=worker,
                owner=local_plan.owner,
            ).wrap()

        def replace_ids(obj, from_id, to_id):
            """
            Takes on object and replace occurrences of change_id with to_id
            :return: the object updated
            """
            if isinstance(obj, (list, tuple)):
                l = [replace_ids(o, from_id, to_id) for o in obj]
                if isinstance(obj, tuple):
                    return tuple(l)
                else:
                    return l
            elif isinstance(obj, int):
                if obj == from_id:
                    return to_id
            elif isinstance(obj, bytes):
                if obj == b"local_plan":
                    return str.encode(f"{worker.id}")
            return obj

        return execute_plan

    return create_plan
