import subprocess


from torchvision import datasets
from torchvision import transforms


# Downloads MNIST dataset
mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

call_alice = ["python", "run_websocket_server.py", "--port", "8777", "--id", "alice"]

call_bob = ["python", "run_websocket_server.py", "--port", "8778", "--id", "bob"]

call_charlie = ["python", "run_websocket_server.py", "--port", "8779", "--id", "charlie"]

print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)
