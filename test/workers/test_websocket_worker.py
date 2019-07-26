import io
from os.path import exists, join
import time
from socket import gethostname
from OpenSSL import crypto, SSL
import pytest
import torch
import syft as sy
from syft.frameworks.torch.federated import utils

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker

PRINT_IN_UNITTESTS = False


def instantiate_websocket_client_worker(**kwargs):  # pragma: no cover
    """ Helper function to instantiate the websocket client.
    If connection is refused, we wait a bit and try again.
    After 5 failed tries, a ConnectionRefusedError is raised.
    """
    retry_counter = 0
    connection_open = False
    while not connection_open:
        try:
            local_worker = WebsocketClientWorker(**kwargs)
            connection_open = True
        except ConnectionRefusedError as e:
            if retry_counter < 5:
                retry_counter += 1
                time.sleep(0.1)
            else:
                raise e
    return local_worker


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
        "id": "secure_fed" if secure else "fed",
        "host": "localhost",
        "port": 8766 if secure else 8765,
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
    local_worker = instantiate_websocket_client_worker(**kwargs)

    x = x.send(local_worker)
    y = x + x
    y = y.get()

    assert (y == torch.ones(5) * 2).all()

    del x

    local_worker.ws.shutdown()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


def test_websocket_workers_search(hook, start_proc):
    """Evaluates that a client can search and find tensors that belong
    to another party"""
    # Sample tensor to store on the server
    sample_data = torch.tensor([1, 2, 3, 4]).tag("#sample_data", "#another_tag")
    # Args for initializing the websocket server and client
    base_kwargs = {"id": "fed2", "host": "localhost", "port": 8767, "hook": hook}
    server_kwargs = base_kwargs
    server_kwargs["data"] = [sample_data]
    process_remote_worker = start_proc(WebsocketServerWorker, **server_kwargs)

    local_worker = instantiate_websocket_client_worker(**base_kwargs)

    # Search for the tensor located on the server by using its tag
    results = local_worker.search("#sample_data", "#another_tag")

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    # Search multiple times should still work
    results = local_worker.search("#sample_data", "#another_tag")

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    local_worker.close()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


def test_list_objects_remote(hook, start_proc):

    kwargs = {"id": "fed", "host": "localhost", "port": 8765, "hook": hook}
    process_remote_fed1 = start_proc(WebsocketServerWorker, **kwargs)
    local_worker = instantiate_websocket_client_worker(**kwargs)

    x = torch.tensor([1, 2, 3]).send(local_worker)

    res = local_worker.list_objects_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 1

    y = torch.tensor([4, 5, 6]).send(local_worker)
    res = local_worker.list_objects_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 2

    # retrieve x and y before terminating the websocket connection
    x.get()
    y.get()
    local_worker.close()
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_fed1.terminate()


def test_objects_count_remote(hook, start_proc):

    kwargs = {"id": "fed", "host": "localhost", "port": 8764, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)
    local_worker = instantiate_websocket_client_worker(**kwargs)

    x = torch.tensor([1, 2, 3]).send(local_worker)

    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 1

    y = torch.tensor([4, 5, 6]).send(local_worker)
    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 2

    x.get()
    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 1

    # get remote object before terminating the websocket connection
    y.get()
    local_worker.close()
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


def test_connect_close(hook, start_proc):
    kwargs = {"id": "fed", "host": "localhost", "port": 8763, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)
    local_worker = instantiate_websocket_client_worker(**kwargs)

    x = torch.tensor([1, 2, 3])
    x_ptr = x.send(local_worker)

    assert local_worker.objects_count_remote() == 1

    local_worker.close()

    time.sleep(0.1)

    local_worker.connect()

    assert local_worker.objects_count_remote() == 1

    x_val = x_ptr.get()
    assert (x_val == x).all()

    local_worker.close()
    local_worker.remove_worker_from_local_worker_registry()
    local_worker.close()

    time.sleep(0.1)

    process_remote_worker.terminate()


def test_websocket_worker_multiple_output_response(hook, start_proc):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker"""

    kwargs = {"id": "socket_multiple_output", "host": "localhost", "port": 8768, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)
    local_worker = instantiate_websocket_client_worker(**kwargs)

    x = torch.tensor([1.0, 3, 2])
    x = x.send(local_worker)
    p1, p2 = torch.sort(x)
    x1, x2 = p1.get(), p2.get()

    assert (x1 == torch.tensor([1.0, 2, 3])).all()
    assert (x2 == torch.tensor([0, 2, 1])).all()

    x.get()  # retrieve remote object before closing the websocket connection

    local_worker.close()
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_evaluate(hook, start_proc):  # pragma: no cover

    sy.local_worker.clear_objects()
    sy.frameworks.torch.hook.hook_args.hook_method_args_functions = {}
    sy.frameworks.torch.hook.hook_args.hook_method_response_functions = {}
    sy.frameworks.torch.hook.hook_args.get_tensor_type_functions = {}
    sy.frameworks.torch.hook.hook_args.register_response_functions = {}

    data, target = utils.iris_data_partial()

    dataset = sy.BaseDataset(data=data, targets=target)

    kwargs = {"id": "evaluate_remote", "host": "localhost", "port": 8780, "hook": hook}
    dataset_key = "iris"
    # TODO: check why unit test sometimes fails when WebsocketServerWorker is started from the unit test. Fails when run after test_federated_client.py
    # process_remote_worker = start_proc(WebsocketServerWorker, dataset=(dataset, dataset_key), verbose=True, **kwargs)

    local_worker = instantiate_websocket_client_worker(**kwargs)

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
        print("Loss: {}".format(loss_before))

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
    train_config.send(local_worker)

    result = local_worker.evaluate(
        dataset_key=dataset_key, calculate_histograms=True, nr_bins=3, calculate_loss=True
    )

    test_loss_before, correct_before, len_dataset, hist_pred_before, hist_target = result

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Evaluation result before training: {}".format(result))

    assert len_dataset == 30
    assert (hist_target == [10, 10, 10]).all()

    local_worker.close()
    local_worker.remove_worker_from_local_worker_registry()
    # process_remote_worker.terminate()
