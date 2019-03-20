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
