# stdlib
import base64
import secrets
from timeit import timeit
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Generator
from typing import List as TypeList
from typing import Union

# third party
import requests
from syft_proto.execution.v1.plan_pb2 import Plan as PlanTorchscriptPB

# syft relative
from ..core.plan import Plan
from ..core.plan.translation.torchscript.plan import PlanTorchscript
from ..federated.model_centric_fl_base import ModelCentricFLBase
from ..lib.python.list import List
from ..proto.core.plan.plan_pb2 import Plan as PlanPB
from .model_serialization import deserialize_model_params
from .model_serialization import wrap_model_params

CHUNK_SIZE = 655360  # 640KB
SPEED_MULT_FACTOR = 10
MAX_BUFFER_SIZE = 1048576 * 64  # 64 MB
MAX_SPEED_TESTS = 3


class ModelCentricFLWorker(ModelCentricFLBase):
    CYCLE_STATUS_ACCEPTED = "accepted"
    CYCLE_STATUS_REJECTED = "rejected"
    PLAN_TYPE_LIST = "list"
    PLAN_TYPE_TORCHSCRIPT = "torchscript"

    def _yield_chunk_from_request(
        self, request: requests.Response, chunk_size: int
    ) -> Generator:
        for chunk in request.iter_content(chunk_size=chunk_size):
            yield chunk

    def _read_n_request_chunks(self, chunk_generator: Generator, n: int) -> bool:
        for i in range(n):
            try:
                next(chunk_generator)
            except Exception:
                return False
        return True

    def _get_ping(self, worker_id: str, random_id: int) -> float:
        params = {"is_ping": 1, "worker_id": worker_id, "random": random_id}
        ping = (
            timeit(
                lambda: self._send_http_req("GET", "/model-centric/speed-test", params),
                number=MAX_SPEED_TESTS,
            )
            * 1000
        )  # for ms
        return ping

    def _get_upload_speed(self, worker_id: str, random_id: int) -> float:
        buffer_size = CHUNK_SIZE
        speed_history = []

        for _ in range(MAX_SPEED_TESTS):
            data_sample = b"x" * buffer_size
            params = {"worker_id": worker_id, "random": random_id}
            body = {"upload_data": data_sample}
            time_taken = timeit(
                lambda: self._send_http_req(
                    "POST", "/model-centric/speed-test", params, body
                ),
                number=1,
            )
            if time_taken < 0.5:
                buffer_size = min(buffer_size * SPEED_MULT_FACTOR, MAX_BUFFER_SIZE)
            new_speed = buffer_size / (time_taken * 1024)

            if new_speed != float("inf"):
                speed_history.append(new_speed)

        if len(speed_history) == 0:
            return float("inf")
        else:
            avg_speed = sum(speed_history) / len(speed_history)
            return avg_speed

    def _get_download_speed(self, worker_id: str, random_id: int) -> float:
        params: TypeDict[str, Union[int, str]] = {
            "worker_id": worker_id,
            "random": random_id,
        }
        speed_history = []
        with requests.get(
            self.http_url + "/model-centric/speed-test", params, stream=True
        ) as r:
            r.raise_for_status()
            buffer_size = CHUNK_SIZE
            chunk_generator = self._yield_chunk_from_request(r, CHUNK_SIZE)
            for _ in range(MAX_SPEED_TESTS):
                time_taken = timeit(
                    lambda: self._read_n_request_chunks(
                        chunk_generator, buffer_size // CHUNK_SIZE
                    ),
                    number=1,
                )
                if time_taken < 0.5:
                    buffer_size = min(buffer_size * SPEED_MULT_FACTOR, MAX_BUFFER_SIZE)
                new_speed = buffer_size / (time_taken * 1024)

                if new_speed != float("inf"):
                    speed_history.append(new_speed)

        if len(speed_history) == 0:
            return float("inf")
        else:
            avg_speed = sum(speed_history) / len(speed_history)
            return avg_speed

    def authenticate(
        self, auth_token: str, model_name: str, model_version: str
    ) -> TypeDict[str, TypeAny]:
        message = {
            "type": "model-centric/authenticate",
            "data": {
                "auth_token": auth_token,
                "model_name": model_name,
                "model_version": model_version,
            },
        }

        return self._send_msg(message)

    def cycle_request(
        self,
        worker_id: str,
        model_name: str,
        model_version: str,
        speed_info: TypeDict[str, TypeAny],
    ) -> TypeDict[str, TypeAny]:
        message = {
            "type": "model-centric/cycle-request",
            "data": {
                "worker_id": worker_id,
                "model": model_name,
                "version": model_version,
                **speed_info,
            },
        }
        return self._send_msg(message)

    def get_model(self, worker_id: str, request_key: str, model_id: int) -> TypeList:
        params_dict = {
            "worker_id": worker_id,
            "request_key": request_key,
            "model_id": model_id,
        }
        serialized_model = self._send_http_req(
            "GET", "/model-centric/get-model", params_dict
        )
        # TODO migrate to syft-core protobufs
        params: List = deserialize_model_params(serialized_model)
        return params.upcast()

    def get_plan(
        self, worker_id: str, request_key: str, plan_id: int, receive_operations_as: str
    ) -> Union[PlanTorchscript, Plan]:
        params = {
            "worker_id": worker_id,
            "request_key": request_key,
            "plan_id": plan_id,
            "receive_operations_as": receive_operations_as,
        }
        serialized_plan = self._send_http_req("GET", "/model-centric/get-plan", params)
        if receive_operations_as == ModelCentricFLWorker.PLAN_TYPE_TORCHSCRIPT:
            # TODO migrate to syft-core protobufs
            pb = PlanTorchscriptPB()
            pb.ParseFromString(serialized_plan)
            return PlanTorchscript._proto2object(pb)
        else:
            return self._unserialize(serialized_plan, PlanPB)

    def report(
        self, worker_id: str, request_key: str, diff: TypeList
    ) -> TypeDict[str, TypeAny]:
        # TODO migrate to syft-core protobufs
        diff_serialized = self._serialize(wrap_model_params(diff))
        diff_base64 = base64.b64encode(diff_serialized).decode("ascii")
        params = {
            "type": "model-centric/report",
            "data": {
                "worker_id": worker_id,
                "request_key": request_key,
                "diff": diff_base64,
            },
        }
        return self._send_msg(params)

    def get_connection_speed(self, worker_id: str) -> TypeDict[str, TypeAny]:
        random_num = secrets.randbits(128)
        ping = self._get_ping(worker_id, random_num)
        upload_speed = self._get_upload_speed(worker_id, random_num)
        download_speed = self._get_download_speed(worker_id, random_num)
        return {"ping": ping, "download": download_speed, "upload": upload_speed}
