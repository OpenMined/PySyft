from datetime import datetime
from random import randint

from src.app.main.model_centric.cycles.cycle import Cycle
from src.app.main.model_centric.models.ai_model import Model, ModelCheckPoint
from src.app.main.model_centric.processes.config import Config
from src.app.main.model_centric.processes.fl_process import FLProcess
from src.app.main.model_centric.syft_assets.plan import Plan
from src.app.main.model_centric.syft_assets.protocol import Protocol

BIG_INT = 2 ** 32

MODELS = [Model(version="0.0.1")]


AVG_PLANS = [
    Plan(
        id=randint(0, BIG_INT),
        value="list of plan values".encode("utf-8"),
        value_ts="torchscript(plan)".encode("utf-8"),
    )
]

TRAINING_PLANS = [
    Plan(
        id=randint(0, BIG_INT),
        value="list of plan values".encode("utf-8"),
        value_ts="torchscript(plan)".encode("utf-8"),
    )
]

VALIDATION_PLANS = [
    Plan(
        id=randint(0, BIG_INT),
        value="list of plan values".encode("utf-8"),
        value_ts="torchscript(plan)".encode("utf-8"),
    )
]


PROTOCOLS = [
    Protocol(
        id=randint(0, BIG_INT),
        value="list of protocol values".encode("utf-8"),
        value_ts="torchscript(protocol)".encode("utf-8"),
    )
]

CLIENT_CONFIGS = [
    Config(
        id=randint(0, BIG_INT),
        config={
            "name": "my-federated-model",
            "version": "0.1.0",
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "SGD",
        },
    )
]


SERVER_CONFIGS = [
    Config(
        id=randint(0, BIG_INT),
        config={
            "max_workers": 100,
            "pool_selection": "random",  # or "iterate"
            "num_cycles": 5,
            "do_not_reuse_workers_until_cycle": 4,
            "cycle_length": 8 * 60 * 60,  # 8 hours
            "minimum_upload_speed": 2000,  # 2 mbps
            "minimum_download_speed": 4000,  # 4 mbps
        },
    )
]

CYCLES = [
    Cycle(
        id=randint(0, BIG_INT),
        start=datetime(2019, 2, 21, 7, 29, 32, 45),
        sequence=randint(0, 100),
        end=datetime(2019, 2, 22, 7, 29, 32, 45),
    )
]
