# stdlib
from typing import Any as TypeAny
from typing import Callable
from typing import Dict as TypeDict
from typing import List as ListType
from typing import Optional

# syft relative
from ..federated import JSONDict
from ..logger import traceback_and_raise
from .model_centric_fl_base import GridError
from .model_centric_fl_worker import ModelCentricFLWorker


class EventEmitter:
    def __init__(self) -> None:
        self.listeners: TypeDict[str, ListType[Callable]] = {}

    def add_listener(self, event_name: str, fn: Callable) -> None:
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(fn)

    def trigger(self, event_name: str, *args: TypeAny, **kwargs: TypeAny) -> None:
        if event_name in self.listeners:
            for fn in self.listeners[event_name]:
                fn(*args, **kwargs)


class FLJob(EventEmitter):
    EVENT_ACCEPTED: str = "accepted"
    EVENT_REJECTED: str = "rejected"
    EVENT_ERROR: str = "error"

    def __init__(
        self,
        worker_id: str,
        grid_worker: ModelCentricFLWorker,
        model_name: str,
        model_version: str,
    ) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.grid_worker = grid_worker
        self.model_name = model_name
        self.model_version = model_version
        self.plan_type = ModelCentricFLWorker.PLAN_TYPE_LIST

        self.model: Optional[ListType[TypeAny]] = None
        self.plans: JSONDict = {}
        self.cycle_params: JSONDict = {}
        self.client_config: JSONDict = {}

    def _init_cycle(self, cycle_params: JSONDict) -> None:
        self.cycle_params = cycle_params
        self.client_config = cycle_params["client_config"]

        request_key = cycle_params["request_key"]

        # Load model
        try:
            self.model = self.grid_worker.get_model(
                self.worker_id, request_key, cycle_params["model_id"]
            )
            if not self.model:
                traceback_and_raise(
                    ValueError(f"Model is not valid so {self} can't run")
                )
        except Exception as e:
            traceback_and_raise(
                ValueError(f"Failed to fetch model during {self}._init_cycle. {e}")
            )

        # Load plans
        for plan_name, plan_id in cycle_params["plans"].items():
            self.plans[plan_name] = self.grid_worker.get_plan(
                self.worker_id,
                request_key,
                plan_id,
                self.plan_type,
            )

    def start(self) -> None:
        try:
            speed_info = self.grid_worker.get_connection_speed(self.worker_id)
            cycle_request_response = self.grid_worker.cycle_request(
                worker_id=self.worker_id,
                model_name=self.model_name,
                model_version=self.model_version,
                speed_info=speed_info,
            )

            cycle_params = cycle_request_response["data"]

            if cycle_params["status"] == ModelCentricFLWorker.CYCLE_STATUS_ACCEPTED:
                self._init_cycle(cycle_params)
                self.trigger(self.EVENT_ACCEPTED, self)
            elif cycle_params["status"] == ModelCentricFLWorker.CYCLE_STATUS_REJECTED:
                timeout = cycle_params.get("timeout")
                if timeout is not None:
                    timeout = timeout(int)
                self.trigger(self.EVENT_REJECTED, self, timeout)
        except GridError as e:
            self.trigger(
                self.EVENT_ERROR,
                self,
                f"Grid communication error: {e.error}",
            )

    def report(self, updated_model_params: ListType) -> JSONDict:
        # Calc params diff
        if not self.model:
            traceback_and_raise(
                ValueError(f"Model is not valid so {self} can't report")
            )
        diff_params: ListType[TypeAny] = [
            o_p - updated_model_params[i] for i, o_p in enumerate(self.model)
        ]

        return self.grid_worker.report(
            worker_id=self.worker_id,
            request_key=self.cycle_params["request_key"],
            diff=diff_params,
        )
