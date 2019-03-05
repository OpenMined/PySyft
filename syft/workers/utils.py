import syft as sy

from .plan import Plan


def make_plan():
    """
    hook args and response for methods that hold the @make_plan decorator
    """

    def create_plan(plan_blueprint, *dummy_args):
        """
        Take as an input a function to be converted into a plan, makes it a plan
        using a new object. The object is still local but it can be sent somewhere
        using the .send() command.

        Args:
            plan_blueprint (func): a function containing PyTorch code.
            dummy_args(list): example arguments for dummy data passed through
                the method.

        """
        args = list(dummy_args)

        plan = Plan(hook=sy.local_worker.hook, owner=sy.local_worker, id="plan")

        # The ids of args of the first call, which should be updated when
        # the function is called with new args
        arg_ids = list()

        for i in range(len(args)):
            args[i] = args[i].send(plan)
            args[i].child.garbage_collect_data = False
            arg_ids.append(args[i].id_at_location)

        plan.arg_ids = arg_ids

        res_ptr = plan_blueprint(*args)

        res_ptr.child.garbage_collect_data = False

        # The id where the result should be stored
        plan.result_ids = [res_ptr.id_at_location]

        return plan

    return create_plan
