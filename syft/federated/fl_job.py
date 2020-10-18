from syft.workers.model_centric_fl_worker import ModelCentricFLWorker
from syft.grid.exceptions import GridError
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder


class EventEmitter:
    def __init__(self):
        self.listeners = {}
        pass

    def add_listener(self, event_name, fn):
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(fn)

    def trigger(self, event_name, *args, **kwargs):
        if event_name in self.listeners:
            for fn in self.listeners[event_name]:
                fn(*args, **kwargs)


class FLJob(EventEmitter):
    EVENT_ACCEPTED = "accepted"
    EVENT_REJECTED = "rejected"
    EVENT_ERROR = "error"

    def __init__(
        self, fl_client, grid_worker: ModelCentricFLWorker, model_name: str, model_version: str
    ):
        super().__init__()
        self.fl_client = fl_client
        self.grid_worker = grid_worker
        self.model_name = model_name
        self.model_version = model_version

        self.model = None
        self.plans = {}
        self.protocols = {}
        self.cycle_params = {}
        self.client_config = {}

    def _init_cycle(self, cycle_params: dict):
        self.cycle_params = cycle_params
        self.client_config = cycle_params["client_config"]

        worker_id = self.fl_client.worker_id
        request_key = cycle_params["request_key"]

        # Load model
        self.model = self.grid_worker.get_model(worker_id, request_key, cycle_params["model_id"])

        # Load plans
        for plan_name, plan_id in cycle_params["plans"].items():
            self.plans[plan_name] = self.grid_worker.get_plan(
                worker_id, request_key, plan_id, ModelCentricFLWorker.PLAN_TYPE_TORCHSCRIPT
            )

        # Load protocols
        for protocol_name, protocol_id in cycle_params["protocols"].items():
            self.protocols[protocol_name] = self.grid_worker.get_protocol(
                worker_id, request_key, protocol_id
            )

    def start(self):
        try:
            speed_info = self.grid_worker.get_connection_speed(self.fl_client.worker_id)
            cycle_request_response = self.grid_worker.cycle_request(
                worker_id=self.fl_client.worker_id,
                model_name=self.model_name,
                model_version=self.model_version,
                speed_info=speed_info,
            )
            cycle_params = cycle_request_response["data"]

            if cycle_params["status"] == ModelCentricFLWorker.CYCLE_STATUS_ACCEPTED:
                self._init_cycle(cycle_params)
                self.trigger(self.EVENT_ACCEPTED, self)
            elif cycle_params["status"] == ModelCentricFLWorker.CYCLE_STATUS_REJECTED:
                self.trigger(self.EVENT_REJECTED, self, cycle_params.get("timeout", None))
        except GridError as e:
            self.trigger(self.EVENT_ERROR, self, f"Grid communication error: {e.error}")

    def report(self, updated_model_params: list):
        # Calc params diff
        orig_params = self.model.tensors()
        diff_params = [orig_params[i] - updated_model_params[i] for i in range(len(orig_params))]

        # Wrap diff in State
        diff_ph = [PlaceHolder().instantiate(t) for t in diff_params]
        diff = State(state_placeholders=diff_ph)

        response = self.grid_worker.report(
            worker_id=self.fl_client.worker_id,
            request_key=self.cycle_params["request_key"],
            diff=diff,
        )
        return response
