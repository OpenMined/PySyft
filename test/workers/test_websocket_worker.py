import io
from os.path import exists, join
import time
from socket import gethostname
from OpenSSL import crypto, SSL
import pytest
import torch

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


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
    local_worker = WebsocketClientWorker(**kwargs)

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

    time.sleep(0.1)

    local_worker = WebsocketClientWorker(**base_kwargs)

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

    local_worker.ws.shutdown()
    local_worker.ws.close()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


def test_list_objects_remote(hook, start_proc):

    kwargs = {"id": "fed", "host": "localhost", "port": 8765, "hook": hook}
    process_remote_fed1 = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)

    kwargs = {"id": "fed", "host": "localhost", "port": 8765, "hook": hook}
    local_worker = WebsocketClientWorker(**kwargs)

    x = torch.tensor([1, 2, 3]).send(local_worker)

    res = local_worker.list_objects_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 1

    y = torch.tensor([4, 5, 6]).send(local_worker)
    res = local_worker.list_objects_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 2

    # delete x before terminating the websocket connection
    del x
    del y
    time.sleep(0.1)
    local_worker.ws.shutdown()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_fed1.terminate()


def test_objects_count_remote(hook, start_proc):

    kwargs = {"id": "fed", "host": "localhost", "port": 8764, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)

    kwargs = {"id": "fed", "host": "localhost", "port": 8764, "hook": hook}
    local_worker = WebsocketClientWorker(**kwargs)

    x = torch.tensor([1, 2, 3]).send(local_worker)

    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 1

    y = torch.tensor([4, 5, 6]).send(local_worker)
    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 2

    x.get()
    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 1

    # delete remote object before terminating the websocket connection
    del y
    time.sleep(0.1)
    local_worker.ws.shutdown()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()


def test_connect_close(hook, start_proc):
    kwargs = {"id": "fed", "host": "localhost", "port": 8763, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)

    kwargs = {"id": "fed", "host": "localhost", "port": 8763, "hook": hook}
    local_worker = WebsocketClientWorker(**kwargs)

    x = torch.tensor([1, 2, 3])
    x_ptr = x.send(local_worker)

    assert local_worker.objects_count_remote() == 1

    local_worker.close()

    time.sleep(0.1)

    local_worker.connect()

    assert local_worker.objects_count_remote() == 1

    x_val = x_ptr.get()
    assert (x_val == x).all()

    local_worker.ws.shutdown()

    time.sleep(0.1)

    process_remote_worker.terminate()


def test_websocket_worker_multiple_output_response(hook, start_proc):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker"""

    kwargs = {"id": "socket_multiple_output", "host": "localhost", "port": 8768, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)
    x = torch.tensor([1.0, 3, 2])

    local_worker = WebsocketClientWorker(**kwargs)

    x = x.send(local_worker)
    p1, p2 = torch.sort(x)
    x1, x2 = p1.get(), p2.get()

    assert (x1 == torch.tensor([1.0, 2, 3])).all()
    assert (x2 == torch.tensor([0, 2, 1])).all()

    x.get()  # retrieve remote object before closing the websocket connection

    local_worker.ws.shutdown()
    process_remote_worker.terminate()
