import torch
import syft as sy
from grid.workers.socketio_server import WebsocketIOServerWorker

# Use Numpy serialization strategy
sy.serde._serialize_tensor = sy.serde.numpy_tensor_serializer
sy.serde._deserialize_tensor = sy.serde.numpy_tensor_deserializer
sy.serde._apply_compress_scheme = sy.serde.apply_no_compression


# Pass this payload when building the WebsocketIOServerWorker instance to make it work as the only component passing
# operations to the other client (i.e. Android)
def _payload(location):
    x = torch.tensor([10, 20, 30, 40, 50.0])
    x_ptr = x.send(location)
    print("x sent")
    print(x_ptr)
    y = torch.tensor([1, 2, 3, 4, 5.0])
    y_ptr = y.send(location)
    print("y sent")
    z_ptr = x_ptr + y_ptr
    print("sum x + y sent")
    z = z_ptr.get()
    print("get sent")
    print(z)
    print("End of payload")


if __name__ == "__main__":
    hook = sy.TorchHook(torch)
    server_worker = WebsocketIOServerWorker(hook, "0.0.0.0", 5000, log_msgs=True)
    server_worker.start()
