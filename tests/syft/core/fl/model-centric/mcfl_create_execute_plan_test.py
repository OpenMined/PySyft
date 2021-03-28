# stdlib
import base64
import json
import os
import time
from typing import Any
from typing import Generator
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple

# third party
import jwt
import numpy as np
import pytest
import requests
import torch as th
from torchvision import datasets
from torchvision import transforms
from websocket import create_connection
from xprocess import ProcessStarter

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan
from syft.federated import JSONDict
from syft.federated.fl_client import FLClient
from syft.federated.fl_job import FLJob
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.lib.python.int import Int
from syft.lib.python.list import List
from syft.lib.torch.module import Module as SyModule
from syft.proto.core.plan.plan_pb2 import Plan as PlanPB
from syft.proto.lib.python.list_pb2 import List as ListPB
from syft.util import get_root_data_path

th.random.manual_seed(42)

here = os.path.dirname(__file__)
DOMAIN_PORT = 7000


@pytest.fixture
def pygrid_domain(xprocess: Any) -> Generator:
    class Starter(ProcessStarter):
        # startup pattern
        pattern = "Starting app"

        # command to start process
        pygrid_path = f"{here}/../../../../../pygrid"
        domain_path = os.path.abspath(f"{pygrid_path}/apps/domain")
        database_file = f"{domain_path}/src/datadomain.db"
        if os.path.exists(database_file):
            os.unlink(database_file)
        args = [
            "python",
            f"{domain_path}/src/__main__.py",
            f"--port={DOMAIN_PORT}",
            "--start_local_db",
        ]

    # ensure process is running and return its logfile
    logfile = xprocess.ensure("pygrid_domain", Starter)

    yield logfile

    # clean up whole process tree afterwards
    xprocess.getinfo("pygrid_domain").terminate()


@pytest.mark.grid
def test_create_and_execute_plan(pygrid_domain: Any) -> None:
    model_param_type_size = create_plan()
    matches = [
        (th.nn.Parameter, th.Size([100, 784])),
        (th.nn.Parameter, th.Size([100])),
        (th.nn.Parameter, th.Size([10, 100])),
        (th.nn.Parameter, th.Size([10])),
    ]

    assert model_param_type_size == matches

    accuracy = execute_plan()
    print(f"Model Centric Federated Learning Complete. Accuracy {accuracy:.2F}")
    assert accuracy > 0.05


class MLP(sy.Module):
    def __init__(self, torch_ref: Any) -> None:
        super().__init__(torch_ref=torch_ref)
        self.l1 = self.torch_ref.nn.Linear(784, 100)
        self.a1 = self.torch_ref.nn.ReLU()
        self.l2 = self.torch_ref.nn.Linear(100, 10)

    def forward(self, x: Any) -> Any:
        x_reshaped = x.view(-1, 28 * 28)
        l1_out = self.a1(self.l1(x_reshaped))
        l2_out = self.l2(l1_out)
        return l2_out


def cross_entropy_loss(
    logits: th.Tensor, targets: th.Tensor, batch_size: int
) -> th.Tensor:
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    return -(targets * log_probs).sum() / batch_size


def sgd_step(model: SyModule, lr: float = 0.1) -> None:
    with ROOT_CLIENT.torch.no_grad():
        for p in model.parameters():
            p.data = p.data - lr * p.grad
            p.grad = th.zeros_like(p.grad.get())


def set_params(model: SyModule, params: List) -> None:
    for p, p_new in zip(model.parameters(), params):
        p.data = p_new.data


def read_file(fname: str) -> str:
    with open(fname, "r") as f:
        return f.read()


private_key = read_file(f"{here}/example_rsa").strip()
public_key = read_file(f"{here}/example_rsa.pub").strip()

auth_token = jwt.encode({}, private_key, algorithm="RS256").decode("ascii")


def create_plan() -> TypeList[TypeTuple[type, th.Size]]:
    local_model = MLP(th)

    @make_plan
    def train(  # type: ignore
        xs=th.rand([64 * 3, 1, 28, 28]),
        ys=th.randint(0, 10, [64 * 3, 10]),
        params=List(local_model.parameters()),
    ):

        model = local_model.send(ROOT_CLIENT)
        set_params(model, params)
        for i in range(1):
            indices = th.tensor(range(64 * i, 64 * (i + 1)))
            x, y = xs.index_select(0, indices), ys.index_select(0, indices)
            out = model(x)
            loss = cross_entropy_loss(out, y, 64)
            loss.backward()
            sgd_step(model)

        return model.parameters()

    @make_plan
    def avg_plan(  # type: ignore
        avg=List(local_model.parameters()),
        item=List(local_model.parameters()),
        num=Int(0),
    ):
        new_avg = []
        for i, param in enumerate(avg):
            new_avg.append((avg[i] * num + item[i]) / (num + 1))
        return new_avg

    name = "mnist"
    version = "1.0"

    client_config = {
        "name": name,
        "version": version,
        "batch_size": 64,
        "lr": 0.1,
        "max_updates": 1,  # custom syft.js option that limits number of training loops per worker
    }

    server_config = {
        "min_workers": 2,
        "max_workers": 2,
        "pool_selection": "random",
        "do_not_reuse_workers_until_cycle": 6,
        "cycle_length": 28800,  # max cycle length in seconds
        "num_cycles": 2,  # max number of cycles
        "max_diffs": 1,  # number of diffs to collect before avg
        "minimum_upload_speed": 0,
        "minimum_download_speed": 0,
        "iterative_plan": True,  # tells PyGrid that avg plan is executed per diff
    }

    server_config["authentication"] = {
        "type": "jwt",
        "pub_key": public_key,
    }

    # Auth
    grid_address = f"localhost:{DOMAIN_PORT}"

    grid = ModelCentricFLClient(address=grid_address, secure=False)
    grid.connect()

    # Host

    # If the process already exists, might you need to clear the db.
    # To do that, set path below correctly and run:
    grid.host_federated_training(
        model=local_model,
        client_plans={"training_plan": train},
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=client_config,
        server_config=server_config,
    )

    # Authenticate for cycle

    # Helper function to make WS requests
    def sendWsMessage(data: JSONDict) -> JSONDict:
        ws = create_connection("ws://" + grid_address)
        ws.send(json.dumps(data))
        message = ws.recv()
        return json.loads(message)

    auth_request = {
        "type": "model-centric/authenticate",
        "data": {
            "model_name": name,
            "model_version": version,
            "auth_token": auth_token,
        },
    }

    auth_response = sendWsMessage(auth_request)

    # Do cycle request

    cycle_request = {
        "type": "model-centric/cycle-request",
        "data": {
            "worker_id": auth_response["data"]["worker_id"],
            "model": name,
            "version": version,
            "ping": 1,
            "download": 10000,
            "upload": 10000,
        },
    }
    cycle_response = sendWsMessage(cycle_request)
    # Download model

    worker_id = auth_response["data"]["worker_id"]
    request_key = cycle_response["data"]["request_key"]
    model_id = cycle_response["data"]["model_id"]
    training_plan_id = cycle_response["data"]["plans"]["training_plan"]

    def get_model(
        grid_address: str, worker_id: str, request_key: str, model_id: int
    ) -> List:
        req = requests.get(
            (
                f"http://{grid_address}/model-centric/get-model?worker_id={worker_id}&"
                f"request_key={request_key}&model_id={model_id}"
            )
        )
        pb = ListPB()
        pb.ParseFromString(req.content)
        return deserialize(pb)

    # Model
    model_params_downloaded = get_model(grid_address, worker_id, request_key, model_id)

    # Download & Execute Plan
    req = requests.get(
        (
            f"http://{grid_address}/model-centric/get-plan?worker_id={worker_id}&"
            f"request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list"
        )
    )
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan = deserialize(pb)

    xs = th.rand([64 * 3, 1, 28, 28])
    ys = th.randint(0, 10, [64 * 3, 10])

    (res,) = plan(xs=xs, ys=ys, params=model_params_downloaded)

    # Report Model diff
    diff = [orig - new for orig, new in zip(res, local_model.parameters())]
    diff_serialized = serialize((List(diff))).SerializeToString()

    params = {
        "type": "model-centric/report",
        "data": {
            "worker_id": worker_id,
            "request_key": request_key,
            "diff": base64.b64encode(diff_serialized).decode("ascii"),
        },
    }

    sendWsMessage(params)

    # Check new model
    req_params = {
        "name": name,
        "version": version,
        "checkpoint": "latest",
    }

    res = requests.get(
        f"http://{grid_address}/model-centric/retrieve-model", req_params
    )

    params_pb = ListPB()
    params_pb.ParseFromString(res.content)
    new_model_params = deserialize(params_pb)

    new_model_params[0]

    param_type_size = [(type(v), v.shape) for v in new_model_params]

    return param_type_size


def execute_plan() -> float:
    # PyGrid Node address
    gridAddress = f"ws://localhost:{DOMAIN_PORT}"

    # Hosted model name/version
    model_name = "mnist"
    model_version = "1.0"

    # syft absolute
    # TorchVision hotfix https://github.com/pytorch/vision/issues/3549

    datasets.MNIST.resources = [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]
    datasets.MNIST(get_root_data_path(), train=True, download=True)
    datasets.MNIST(get_root_data_path(), train=False, download=True)

    tfs = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        get_root_data_path(), train=True, download=True, transform=tfs
    )

    cycles_log: TypeList = []
    status = {"ended": False}

    # Called when client is accepted into FL cycle
    def on_accepted(job: FLJob) -> None:
        print(f"Accepted into {job} cycle {len(cycles_log) + 1}.")

        cycle_params = job.client_config
        batch_size, max_updates = (
            cycle_params["batch_size"],
            cycle_params["max_updates"],
        )
        training_plan, model_params = job.plans["training_plan"], job.model
        losses: TypeList = []
        accuracies: TypeList = []

        train_loader = th.utils.data.DataLoader(
            train_set, batch_size=batch_size, drop_last=True, shuffle=True
        )

        for batch_idx, (x, y) in enumerate(train_loader):
            y = th.nn.functional.one_hot(y, 10)
            (model_params,) = training_plan(xs=x, ys=y, params=model_params)

            if batch_idx >= max_updates - 1:
                break

        job.report(model_params)

        # Save losses/accuracies from cycle
        cycles_log.append((losses, accuracies))

    # Called when the client is rejected from cycle
    def on_rejected(job: FLJob, timeout: Optional[int] = None) -> None:
        if timeout is None:
            print(f"Rejected from {job} cycle without timeout, FL training complete.")
        else:
            print(f"Rejected from {job} cycle with timeout: {timeout}.")
        status["ended"] = True

    # Called when error occurred
    def on_error(job: FLJob, error: Exception) -> None:
        print(f"Error: {job} {error}")
        status["ended"] = True

    def create_client_and_run_cycle() -> None:
        client = FLClient(url=gridAddress, auth_token=auth_token, secure=False)
        client.worker_id = client.grid_worker.authenticate(
            client.auth_token, model_name, model_version
        )["data"]["worker_id"]
        job = client.new_job(model_name, model_version)

        # Set event handlers
        job.add_listener(job.EVENT_ACCEPTED, on_accepted)
        job.add_listener(job.EVENT_REJECTED, on_rejected)
        job.add_listener(job.EVENT_ERROR, on_error)

        # Shoot!
        job.start()

    while not status["ended"]:
        create_client_and_run_cycle()
        time.sleep(1)

    # Download trained model
    grid_address = f"localhost:{DOMAIN_PORT}"
    grid = ModelCentricFLClient(address=grid_address, secure=False)
    grid.connect()

    trained_params = grid.retrieve_model(model_name, model_version)
    # Inference

    def test(test_loader: th.utils.data.DataLoader, model: SyModule) -> th.Tensor:
        correct = []
        model.eval()
        for data, target in test_loader:
            output = model(data)
            _, pred = th.max(output, 1)
            correct.append(th.sum(np.squeeze(pred.eq(target.data.view_as(pred)))))
        acc = sum(correct) / len(test_loader.dataset)
        return acc

    model = MLP(th)
    set_params(model, trained_params)

    tfs = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_test = datasets.MNIST(get_root_data_path(), train=False, transform=tfs)
    test_loader = th.utils.data.DataLoader(
        mnist_test, batch_size=32, shuffle=True, pin_memory=True
    )

    accuracy = test(test_loader, model)

    return accuracy.item()
