import random

import syft as sy
from syft.workers.plan import Plan


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


def meth2plan(plan_blueprint):
    """
    the @meth2plan decorator - converts a method containing sequential pytorch code into
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
