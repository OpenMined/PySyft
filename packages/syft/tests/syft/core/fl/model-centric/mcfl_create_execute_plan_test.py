# stdlib
import base64
from collections import OrderedDict
import json
import os
import time
from typing import Any
from typing import Dict as TypeDict
from typing import Generator
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# third party
import jwt
import numpy as np
import pytest
import requests
from syft_proto.execution.v1.plan_pb2 import Plan as PlanTorchscriptPB
import torch as th
from torchvision import datasets
from torchvision import transforms
from websocket import create_connection
from xprocess import ProcessStarter

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize
from syft.core.plan import Plan
from syft.core.plan.plan_builder import PLAN_BUILDER_VM
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan
from syft.core.plan.translation.torchscript.plan import PlanTorchscript
from syft.core.plan.translation.torchscript.plan_translate import (
    translate as translate_to_ts,
)
from syft.federated import JSONDict
from syft.federated.fl_client import FLClient
from syft.federated.fl_job import FLJob
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.federated.model_serialization import deserialize_model_params
from syft.federated.model_serialization import wrap_model_params
from syft.grid.client.client import connect
from syft.lib.python.int import Int
from syft.lib.python.list import List
from syft.lib.torch.module import Module as SyModule
from syft.proto.core.plan.plan_pb2 import Plan as PlanPB
from syft.util import get_root_data_path

th.random.manual_seed(42)

here = os.path.dirname(__file__)
DOMAIN_PORT = 7000


def setup_domain() -> None:
    # this ensures that the new PyGrid Domain is setup and will respond to commands
    try:
        ua_client = connect(url=f"http://localhost:{DOMAIN_PORT}")

        ua_client.setup(
            domain_name="OpenMined Domain",
            email="owner@myorg.com",
            password="ownerpwd",
            token="9G9MJ06OQH",
        )

    except Exception as e:
        if "domain already has an owner" not in str(e):
            raise e
        else:
            print(f"Failed to run setup. {e}")


@pytest.fixture
def pygrid_domain(xprocess: Any) -> Generator:
    class Starter(ProcessStarter):
        # startup pattern
        pattern = "Starting app"

        # command to start process
        pygrid_path = os.environ.get(
            "TEST_PYGRID_PATH", f"{here}/../../../../../../grid"
        )
        domain_path = os.path.abspath(f"{pygrid_path}/apps/domain")
        database_file = f"{domain_path}/src/nodedatabase.db"
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
def test_create_and_execute_plan_autograd(pygrid_domain: Any) -> None:
    setup_domain()

    fl_name = "mnist_autograd"
    plan, model = create_plan_autograd()

    plans = {"training_plan": plan}
    host_to_grid(plans, model, fl_name)

    bs = 64 * 3
    plan_inputs = OrderedDict(
        {
            "xs": th.rand([bs, 28 * 28]),
            "ys": th.nn.functional.one_hot(th.randint(0, 10, [bs]), 10),
        }
    )
    plan_params_output_idx = [0, 1, 2, 3]
    model_param_type_size = sanity_check_hosted_plan(
        fl_name, model, plan_inputs, plan_params_output_idx
    )
    matches = [
        (th.Tensor, th.Size([100, 784])),
        (th.Tensor, th.Size([100])),
        (th.Tensor, th.Size([10, 100])),
        (th.Tensor, th.Size([10])),
    ]

    assert model_param_type_size == matches

    train_with_hosted_training_plan(fl_name, OrderedDict(), plan_params_output_idx)
    accuracy = check_resulting_model(fl_name, model)

    print(f"Model Centric Federated Learning Complete. Accuracy {accuracy:.2F}")
    assert accuracy > 0.05


@pytest.mark.grid
@pytest.mark.parametrize("plan_type", ["list", "torchscript"])
def test_create_and_execute_plan_mobile(pygrid_domain: Any, plan_type: str) -> None:
    setup_domain()

    fl_name = "mnist_mobile"
    plan, plan_ts, model = create_plan_mobile()

    plans = {
        "training_plan": plan,
        "training_plan:ts": plan_ts,
    }
    host_to_grid(plans, model, fl_name)

    bs = 64
    classes_num = 10
    plan_inputs = OrderedDict(
        {
            "xs": th.rand(bs, 28 * 28),
            "ys": th.nn.functional.one_hot(
                th.randint(0, classes_num, [bs]), classes_num
            ),
            "batch_size": th.tensor([bs]),
            "lr": th.tensor([0.1]),
        }
    )
    plan_output_params_idx = [2, 3, 4, 5]

    model_param_type_size = sanity_check_hosted_plan(
        fl_name, model, plan_inputs, plan_output_params_idx, plan_type
    )
    matches = [
        (th.Tensor, th.Size([100, 784])),
        (th.Tensor, th.Size([100])),
        (th.Tensor, th.Size([10, 100])),
        (th.Tensor, th.Size([10])),
    ]
    assert model_param_type_size == matches

    train_with_hosted_training_plan(
        fl_name, plan_inputs, plan_output_params_idx, plan_type
    )
    accuracy = check_resulting_model(fl_name, model)

    print(f"Model Centric Federated Learning Complete. Accuracy {accuracy:.2F}")
    assert accuracy > 0.05


# === Autograd Plan ===


class MLPAutograd(sy.Module):
    def __init__(self, torch_ref: Any) -> None:
        super().__init__(torch_ref=torch_ref)
        self.l1 = self.torch_ref.nn.Linear(784, 100)
        self.a1 = self.torch_ref.nn.ReLU()
        self.l2 = self.torch_ref.nn.Linear(100, 10)

    def forward(self, x: Any) -> Any:
        l1_out = self.a1(self.l1(x))
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


# === Non-Autograd Plan ===


class MLPNoAutograd(sy.Module):
    """
    Simple model with method for loss and hand-written backprop.
    """

    def __init__(self, torch_ref: Any = th) -> None:
        super(MLPNoAutograd, self).__init__(torch_ref=torch_ref)
        self.fc1 = torch_ref.nn.Linear(784, 100)
        self.relu = torch_ref.nn.ReLU()
        self.fc2 = torch_ref.nn.Linear(100, 10)

    def forward(self, x: Any) -> Any:
        self.z1 = self.fc1(x)
        self.a1 = self.relu(self.z1)
        return self.fc2(self.a1)

    def backward(self, X: Any, error: Any) -> TypeTuple[Any, ...]:
        z1_grad = (error @ self.fc2.state_dict()["weight"]) * (self.a1 > 0).float()
        fc1_weight_grad = z1_grad.t() @ X
        fc1_bias_grad = z1_grad.sum(0)
        fc2_weight_grad = error.t() @ self.a1
        fc2_bias_grad = error.sum(0)
        return fc1_weight_grad, fc1_bias_grad, fc2_weight_grad, fc2_bias_grad

    def softmax_cross_entropy_with_logits(
        self, logits: Any, target: Any, batch_size: int
    ) -> TypeTuple[Any, ...]:
        probs = self.torch_ref.softmax(logits, dim=1)
        loss = -(target * self.torch_ref.log(probs)).sum(dim=1).mean()
        loss_grad = (probs - target) / batch_size
        return loss, loss_grad

    def accuracy(self, logits: Any, targets: Any, batch_size: int) -> Any:
        pred = self.torch_ref.argmax(logits, dim=1)
        targets_idx = self.torch_ref.argmax(targets, dim=1)
        acc = pred.eq(targets_idx).sum().float() / batch_size
        return acc


def read_file(fname: str) -> str:
    with open(fname, "r") as f:
        return f.read()


private_key = read_file(f"{here}/example_rsa").strip()
public_key = read_file(f"{here}/example_rsa.pub").strip()

auth_token = jwt.encode({}, private_key, algorithm="RS256").decode("ascii")


def create_plan_autograd() -> TypeTuple[Plan, SyModule]:
    local_model = MLPAutograd(th)

    @make_plan
    def train(  # type: ignore
        xs=th.rand([64 * 3, 28 * 28]),
        ys=th.nn.functional.one_hot(th.randint(0, 10, [64 * 3]), 10),
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

        return (*model.parameters(),)

    return train, local_model


def create_plan_mobile() -> Any:
    def set_remote_model_params(module_ptrs, params_list_ptr):  # type: ignore
        param_idx = 0
        for module_name, module_ptr in module_ptrs.items():
            for param_name, _ in PLAN_BUILDER_VM.store[
                module_ptr.id_at_location
            ].data.named_parameters():
                module_ptr.register_parameter(param_name, params_list_ptr[param_idx])
                param_idx += 1

    local_model = MLPNoAutograd(th)

    # Dummy inputs
    bs = 3
    classes_num = 10
    model_params_zeros = sy.lib.python.List(
        [th.nn.Parameter(th.zeros_like(param)) for param in local_model.parameters()]
    )

    @make_plan
    def training_plan(  # type: ignore
        xs=th.randn(bs, 28 * 28),
        ys=th.nn.functional.one_hot(th.randint(0, classes_num, [bs]), classes_num),
        batch_size=th.tensor([bs]),
        lr=th.tensor([0.1]),
        params=model_params_zeros,
    ):
        # send the model to plan builder (but not its default params)
        model = local_model.send(ROOT_CLIENT, send_parameters=False)

        # set model params from input
        set_remote_model_params(model.modules, params)

        # forward
        logits = model(xs)

        # loss
        loss, loss_grad = model.softmax_cross_entropy_with_logits(
            logits, ys, batch_size
        )

        # backward
        grads = model.backward(xs, loss_grad)

        # SGD step
        updated_params = tuple(
            param - lr * grad for param, grad in zip(model.parameters(), grads)
        )

        # accuracy
        acc = model.accuracy(logits, ys, batch_size)

        # return things
        return (loss, acc, *updated_params)

    # Translate to torchscript
    ts_plan = translate_to_ts(training_plan)

    return training_plan, ts_plan, local_model


def host_to_grid(plans: TypeDict, model: SyModule, name: str) -> None:
    @make_plan
    def avg_plan(  # type: ignore
        avg=List(model.parameters()),
        item=List(model.parameters()),
        num=Int(0),
    ):
        new_avg = []
        for i, param in enumerate(avg):
            new_avg.append((avg[i] * num + item[i]) / (num + 1))
        return new_avg

    name = name

    client_config = {
        "name": name,
        "version": "1.0",
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
        "num_cycles": 4,  # max number of cycles
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
    grid.host_federated_training(
        model=model,
        client_plans=plans,
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=client_config,
        server_config=server_config,
    )


def sanity_check_hosted_plan(
    name: str,
    model: SyModule,
    plan_inputs: OrderedDict,
    plan_output_params_idx: TypeList[int],
    plan_type: str = "list",
) -> TypeList[TypeTuple[type, th.Size]]:
    grid_address = f"localhost:{DOMAIN_PORT}"
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
            "model_version": "1.0",
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
            "version": "1.0",
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
        # TODO migrate to syft-core protobufs
        return deserialize_model_params(req.content)

    # Model
    model_params_downloaded = get_model(grid_address, worker_id, request_key, model_id)

    def get_plan(
        grid_address: str,
        worker_id: int,
        request_key: str,
        plan_id: int,
        plan_type: str,
    ) -> TypeUnion[PlanTorchscript, Plan]:
        req = requests.get(
            (
                f"http://{grid_address}/model-centric/get-plan?worker_id={worker_id}&"
                f"request_key={request_key}&plan_id={plan_id}&receive_operations_as={plan_type}"
            )
        )

        if plan_type == "torchscript":
            pb = PlanTorchscriptPB()
            pb.ParseFromString(req.content)
            return PlanTorchscript._proto2object(pb)
        else:
            pb = PlanPB()
            pb.ParseFromString(req.content)
            return deserialize(pb)

    # Download & Execute Plan
    plan = get_plan(grid_address, worker_id, request_key, training_plan_id, plan_type)
    plan_inputs["params"] = [
        th.nn.Parameter(param) for param in model_params_downloaded
    ]

    if plan_type == "torchscript":
        # kwargs are not supported in torchscript plan
        res = plan(*plan_inputs.values())
    else:
        res = plan(**plan_inputs)

    updated_params = [res[idx] for idx in plan_output_params_idx]

    # Report Model diff
    diff = [orig - new for new, orig in zip(updated_params, model.parameters())]
    diff_serialized = serialize(wrap_model_params(diff)).SerializeToString()

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
        "version": "1.0",
        "checkpoint": "latest",
    }

    res = requests.get(
        f"http://{grid_address}/model-centric/retrieve-model", req_params
    )

    new_model_params = deserialize_model_params(res.content)
    param_type_size = [(type(v), v.shape) for v in new_model_params]

    return param_type_size


def train_with_hosted_training_plan(
    name: str,
    plan_inputs: OrderedDict,
    plan_output_params_idx: TypeList[int],
    plan_type: str = "list",
) -> None:
    # PyGrid Node address
    gridAddress = f"ws://localhost:{DOMAIN_PORT}"

    # Hosted model name/version
    model_name = name
    model_version = "1.0"

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
            x = x.view(-1, 28 * 28)
            y = th.nn.functional.one_hot(y, 10)
            inputs = plan_inputs
            inputs["xs"] = x
            inputs["ys"] = y
            inputs["params"] = [th.nn.Parameter(param) for param in model_params]
            if plan_type == "torchscript":
                res = training_plan(*inputs.values())
            else:
                res = training_plan(**inputs)
            model_params = [res[idx] for idx in plan_output_params_idx]

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

        # Override plan type to use
        job.plan_type = plan_type

        # Set event handlers
        job.add_listener(job.EVENT_ACCEPTED, on_accepted)
        job.add_listener(job.EVENT_REJECTED, on_rejected)
        job.add_listener(job.EVENT_ERROR, on_error)

        # Shoot!
        job.start()

    while not status["ended"]:
        create_client_and_run_cycle()
        time.sleep(1)


def check_resulting_model(name: str, model: SyModule) -> float:
    # Download trained model
    grid_address = f"localhost:{DOMAIN_PORT}"
    grid = ModelCentricFLClient(address=grid_address, secure=False)
    grid.connect()
    trained_params = grid.retrieve_model(name, "1.0")

    # Inference
    def test(test_loader: th.utils.data.DataLoader, model: SyModule) -> th.Tensor:
        correct = []
        model.eval()
        for data, target in test_loader:
            x = data.view(-1, 28 * 28)
            output = model(x)
            _, pred = th.max(output, 1)
            correct.append(th.sum(np.squeeze(pred.eq(target.data.view_as(pred)))))
        acc = sum(correct) / len(test_loader.dataset)
        return acc

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
