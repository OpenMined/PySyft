import io
from os.path import exists, join
import time
from socket import gethostname
from OpenSSL import crypto, SSL
import pytest
import torch
import syft as sy
from syft.generic.frameworks.hook import hook_args
from syft.frameworks.torch.fl import utils

from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker

from test.conftest import instantiate_websocket_client_worker


PRINT_IN_UNITTESTS = False


@pytest.mark.parametrize("secure", [True, False])
def test_websocket_worker_basic(hook, start_proc, secure, tmpdir):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker in insecure and secure mode."""

    def create_self_signed_cert(cert_path, key_path):
        # create a key pair
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 1024)

        # create a self-signed cert
        cert = crypto.X509()
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(1000)
        cert.set_pubkey(k)
        cert.sign(k, "sha1")

        # store keys and cert
        open(cert_path, "wb").write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        open(key_path, "wb").write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

    kwargs = {
        "id": "secure_fed" if secure else "not_secure_fed",
        "host": "localhost",
        "port": 8766,
        "hook": hook,
    }

    if secure:
        # Create cert and keys
        cert_path = tmpdir.join("test.crt")
        key_path = tmpdir.join("test.key")
        create_self_signed_cert(cert_path, key_path)
        kwargs["cert_path"] = cert_path
        kwargs["key_path"] = key_path

    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)
    x = torch.ones(5)

    if secure:
        # unused args
        del kwargs["cert_path"]
        del kwargs["key_path"]

    kwargs["secure"] = secure
    remote_proxy = instantiate_websocket_client_worker(**kwargs)

    x = x.send(remote_proxy)
    y = x + x
    y = y.get()

    assert (y == torch.ones(5) * 2).all()

    del x

    remote_proxy.close()
    time.sleep(0.1)
    remote_proxy.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


def test_websocket_workers_search(hook, start_remote_worker):
    """Evaluates that a client can search and find tensors that belong
    to another party"""
    # Args for initializing the websocket server and client
    server, remote_proxy = start_remote_worker(id="fed2", hook=hook, port=8767)

    # Sample tensor to store on the server
    sample_data = torch.tensor([1, 2, 3, 4]).tag("#sample_data", "#another_tag")
    _ = sample_data.send(remote_proxy)

    # Search for the tensor located on the server by using its tag
    results = remote_proxy.search(["#sample_data", "#another_tag"])

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    # Search multiple times should still work
    results = remote_proxy.search(["#sample_data", "#another_tag"])

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    remote_proxy.close()
    time.sleep(0.1)
    remote_proxy.remove_worker_from_local_worker_registry()
    server.terminate()


def test_list_objects_remote(hook, start_remote_worker):
    server, remote_proxy = start_remote_worker(id="fed-list-objects", hook=hook, port=8765)
    remote_proxy.clear_objects()

    x = torch.tensor([1, 2, 3]).send(remote_proxy)

    res = remote_proxy.list_tensors_remote()

    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 1

    y = torch.tensor([4, 5, 6]).send(remote_proxy)
    res = remote_proxy.list_tensors_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 2

    # delete x before terminating the websocket connection
    del x
    del y
    time.sleep(0.1)
    remote_proxy.close()
    time.sleep(0.1)
    remote_proxy.remove_worker_from_local_worker_registry()
    server.terminate()


def test_objects_count_remote(hook, start_remote_worker):
    server, remote_proxy = start_remote_worker(id="fed-count-objects", hook=hook, port=8764)
    remote_proxy.clear_objects()

    x = torch.tensor([1, 2, 3]).send(remote_proxy)

    nr_objects = remote_proxy.tensors_count_remote()
    assert nr_objects == 1

    y = torch.tensor([4, 5, 6]).send(remote_proxy)
    nr_objects = remote_proxy.tensors_count_remote()
    assert nr_objects == 2

    x.get()
    nr_objects = remote_proxy.tensors_count_remote()
    assert nr_objects == 1

    # delete remote object before terminating the websocket connection
    del y
    time.sleep(0.1)
    remote_proxy.close()
    time.sleep(0.1)
    remote_proxy.remove_worker_from_local_worker_registry()
    server.terminate()


def test_clear_objects_remote(hook, start_remote_worker):
    server, remote_proxy = start_remote_worker(id="fed-clear-objects", hook=hook, port=8769)

    x = torch.tensor([1, 2, 3]).send(remote_proxy, garbage_collect_data=False)
    y = torch.tensor(4).send(remote_proxy, garbage_collect_data=False)

    nr_objects = remote_proxy.tensors_count_remote()
    assert nr_objects == 2

    remote_proxy.clear_objects_remote()
    nr_objects = remote_proxy.objects_count_remote()
    assert nr_objects == 0

    remote_proxy.close()
    remote_proxy.remove_worker_from_local_worker_registry()
    server.terminate()


def test_connect_close(hook, start_remote_worker):
    server, remote_proxy = start_remote_worker(id="fed-connect-close", hook=hook, port=8770)

    x = torch.tensor([1, 2, 3])
    x_ptr = x.send(remote_proxy)

    assert remote_proxy.tensors_count_remote() == 1

    remote_proxy.close()

    time.sleep(0.1)

    remote_proxy.connect()

    assert remote_proxy.tensors_count_remote() == 1

    x_val = x_ptr.get()
    assert (x_val == x).all()

    remote_proxy.close()
    remote_proxy.remove_worker_from_local_worker_registry()

    time.sleep(0.1)

    server.terminate()


def test_websocket_worker_multiple_output_response(hook, start_remote_worker):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker."""
    server, remote_proxy = start_remote_worker(id="socket_multiple_output", hook=hook, port=8771)

    x = torch.tensor([1.0, 3, 2])
    x = x.send(remote_proxy)

    p1, p2 = torch.sort(x)
    x1, x2 = p1.get(), p2.get()

    assert (x1 == torch.tensor([1.0, 2, 3])).all()
    assert (x2 == torch.tensor([0, 2, 1])).all()

    x.get()  # retrieve remote object before closing the websocket connection

    remote_proxy.close()
    server.terminate()


def test_send_command_whitelist(hook, start_remote_worker):
    server, remote_proxy = start_remote_worker(
        id="worker_call_api_good_methods", hook=hook, port=8772
    )
    whitelisted_methods = {
        "torch": {"tensor": [1, 2, 3], "rand": (2, 3), "randn": (2, 3), "zeros": (2, 3)}
    }

    for framework, methods in whitelisted_methods.items():
        attr = getattr(remote_proxy.remote, framework)

        for method, inp in methods.items():
            x = getattr(attr, method)(inp)

            if "rand" not in method:
                assert (x.get() == getattr(torch, method)(inp)).all()

    remote_proxy.close()
    server.terminate()


def test_send_command_not_whitelisted(hook, start_remote_worker):
    server, remote_proxy = start_remote_worker(
        id="worker_call_api_bad_method", hook=hook, port=8773
    )

    method_not_exist = "openmind"

    for framework in remote_proxy.remote.frameworks:
        if framework in dir(remote_proxy.remote):
            attr = getattr(remote_proxy.remote, framework)
            with pytest.raises(AttributeError):
                getattr(attr, method_not_exist)

    remote_proxy.close()
    server.terminate()


@pytest.mark.skip
def test_evaluate(hook, start_proc):  # pragma: no cover

    sy.local_worker.clear_objects()
    sy.generic.frameworks.hook.hook_args.hook_method_args_functions = {}
    sy.generic.frameworks.hook.hook_args.hook_method_response_functions = {}
    sy.generic.frameworks.hook.hook_args.get_tensor_type_functions = {}
    sy.generic.frameworks.hook.hook_args.register_response_functions = {}

    data, target = utils.iris_data_partial()

    dataset = sy.BaseDataset(data=data, targets=target)

    kwargs = {"id": "evaluate_remote", "host": "localhost", "port": 8790, "hook": hook}
    dataset_key = "iris"
    # TODO: check why unit test sometimes fails when WebsocketServerWorker is started from the unit test. Fails when run after test_federated_client.py
    # process_remote_worker = start_proc(WebsocketServerWorker, dataset=(dataset, dataset_key), verbose=True, **kwargs)

    remote_proxy = instantiate_websocket_client_worker(**kwargs)

    def loss_fn(pred, target):
        return torch.nn.functional.cross_entropy(input=pred, target=target)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(4, 3)

            torch.nn.init.xavier_normal_(self.fc1.weight)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            return x

    model_untraced = Net()
    model = torch.jit.trace(model_untraced, data)
    loss_traced = torch.jit.trace(loss_fn, (torch.tensor([[0.3, 0.5, 0.2]]), torch.tensor([1])))

    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print(f"Loss: {loss_before}")

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=4,
        model=model,
        loss_fn=loss_traced,
        model_id=None,
        loss_fn_id=None,
        optimizer_args=None,
        epochs=1,
    )
    train_config.send(remote_proxy)

    result = remote_proxy.evaluate(
        dataset_key=dataset_key, return_histograms=True, nr_bins=3, return_loss=True
    )

    len_dataset = result["nr_predictions"]
    hist_target = result["histogram_target"]

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print(f"Evaluation result before training: {result}")

    assert len_dataset == 30
    assert (hist_target == [10, 10, 10]).all()

    remote_proxy.close()
    remote_proxy.remove_worker_from_local_worker_registry()
    # process_remote_worker.terminate()
