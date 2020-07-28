import pytest
import sys

from . import BIG_INT

from .presets.fl_process import (
    MODELS,
    AVG_PLANS,
    TRAINING_PLANS,
    VALIDATION_PLANS,
    PROTOCOLS,
    CLIENT_CONFIGS,
    SERVER_CONFIGS,
    CYCLES,
)
from .presets.worker_cycle import WORKERS

from random import randint

from src.app.main.sfl.processes.fl_process import FLProcess
from src.app.main.sfl.cycles.worker_cycle import WorkerCycle

sys.path.append(".")


@pytest.mark.parametrize(
    """model,
                         avg_plan,
                         train_plan,
                         valid_plan,
                         protocol,
                         client_config,
                         server_config,
                         cycle,
                         worker""",
    list(
        zip(
            MODELS,
            AVG_PLANS,
            TRAINING_PLANS,
            VALIDATION_PLANS,
            PROTOCOLS,
            CLIENT_CONFIGS,
            SERVER_CONFIGS,
            CYCLES,
            WORKERS,
        )
    ),
)
def test_create_worker_cycles_objects(
    model,
    avg_plan,
    train_plan,
    valid_plan,
    protocol,
    client_config,
    server_config,
    cycle,
    worker,
    database,
):
    new_fl_process = FLProcess(id=randint(0, BIG_INT))

    database.session.add(new_fl_process)

    model.flprocess = new_fl_process
    database.session.add(model)

    avg_plan.avg_flprocess = new_fl_process
    database.session.add(avg_plan)

    train_plan.plan_flprocess = new_fl_process
    database.session.add(train_plan)

    valid_plan.plan_flprocess = new_fl_process
    database.session.add(valid_plan)

    protocol.protocol_flprocess = new_fl_process
    database.session.add(protocol)

    client_config.client_flprocess_config = new_fl_process
    database.session.add(client_config)

    server_config.server_flprocess_config = new_fl_process
    database.session.add(server_config)

    cycle.cycle_flprocess = new_fl_process
    database.session.add(cycle)

    worker_cycle = WorkerCycle(
        id=randint(0, BIG_INT),
        request_key="long_hashcode_here",
        worker=worker,
        cycle=cycle,
    )

    database.session.add(worker_cycle)
    database.session.commit()
