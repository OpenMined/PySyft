import syft as sy
import random
from .plan import Plan


def make_plan(plan_blueprint):
    """For folks who would prefer to not use a decorator, they can use this function"""
    return func2plan()(plan_blueprint)


def func2plan():
    """
    the @func2plan decorator - converts a function of pytorch code into a plan object
    which can be sent to any arbitrary worker.
    """

    def create_plan(plan_blueprint):
        """
        Take as an input a function to be converted into a plan, makes it a plan
        using a new object. The object is still local but it can be sent somewhere
        using the .send() command.

        Args:
            plan_blueprint (func): a function containing PyTorch code.
            dummy_args(list): example arguments for dummy data passed through
                the method.

        """

        plan = Plan(
            hook=sy.local_worker.hook,
            owner=sy.local_worker,
            id=random.randint(0, 1e10),
            name=plan_blueprint.__name__,
        )

        plan.plan_blueprint = plan_blueprint

        return plan

    return create_plan
