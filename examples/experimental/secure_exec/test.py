import crypten
import torch
import torch.nn as nn
import torch.nn.functional as F
import syft
from syft import WebsocketClientWorker
from syft.frameworks.crypten.context import run_multiworkers


# RUN './mnist_utils.py --option features --reduced 100 --binary' first

# TODO
# - Implement classes inside crypten function

hook = syft.TorchHook(torch)

# Syft workers
print("[%] Connecting to workers ...")
LOCAL = syft.local_worker
# ALICE = syft.VirtualWorker(hook=hook, id="alice")
ALICE = WebsocketClientWorker(hook=hook, id="alice", host="127.0.0.1", port=8777)
# BOB = syft.VirtualWorker(hook=hook, id="bob")
BOB = WebsocketClientWorker(hook=hook, id="bob", host="127.0.0.1", port=8778)
print("[+] Connected to workers")

print("[%] Sending labels and training data ...")
# Prepare and send labels
label_eye = torch.eye(2)
labels = torch.load("/tmp/train_labels.pth")
labels = labels.long()
labels_one_hot = label_eye[labels]
labels_one_hot.tag("labels")
ll_ptr = labels_one_hot.send(LOCAL)
al_ptr = labels_one_hot.send(ALICE)
bl_ptr = labels_one_hot.send(BOB)

# Prepare and send training data
alice_train = torch.load("/tmp/alice_train.pth").tag("alice_train")
at_ptr = alice_train.send(ALICE)
bob_train = torch.load("/tmp/bob_train.pth").tag("bob_train")
bt_ptr = bob_train.send(BOB)

print("[+] Data ready")


@run_multiworkers([ALICE, BOB], master_addr="127.0.0.1")
def run_encrypted_training():
    rank = crypten.communicator.get().get_rank()
    # Convert labels to one-hot encoding
    if rank == 0:  # Local party
        labels_one_hot = syft.local_worker.search("labels")[0].get()
    elif rank == 1:  # alice
        worker = syft.local_worker.get_worker("alice")
        labels_one_hot = worker.search("labels")[0].get()
    elif rank == 2:  # bob
        worker = syft.local_worker.get_worker("bob")
        labels_one_hot = worker.search("labels")[0].get()

    # Load data:
    x_alice_enc = crypten.load("alice_train", 1, "alice")
    x_bob_enc = crypten.load("bob_train", 2, "bob")

    # Combine the feature sets: identical to Tutorial 3
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)

    # Reshape to match the network architecture
    x_combined_enc = x_combined_enc.unsqueeze(1)

    # Initialize a plaintext model and convert to CrypTen model
    dummy_input = torch.empty(1, 1, 28, 28)
    model = crypten.nn.from_pytorch(ExampleNet, dummy_input)  # noqa: F821
    model.encrypt()

    # Set train mode
    model.train()

    # Define a loss function
    loss = crypten.nn.MSELoss()

    # Define training parameters
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 10
    num_batches = x_combined_enc.size(0) // batch_size

    for i in range(num_epochs):
        # Print once for readability
        if rank == 0:
            print(f"Epoch {i} in progress:")
            pass

        for batch in range(num_batches):
            # define the start and end of the training mini-batch
            start, end = batch * batch_size, (batch + 1) * batch_size

            # construct AutogradCrypTensors out of training examples / labels
            x_train = crypten.autograd_cryptensor.AutogradCrypTensor(x_combined_enc[start:end])
            y_batch = labels_one_hot[start:end]
            y_train = crypten.autograd_cryptensor.AutogradCrypTensor(crypten.cryptensor(y_batch))

            # perform forward pass:
            output = model(x_train)
            loss_value = loss(output, y_train)

            # set gradients to "zero"
            model.zero_grad()

            # perform backward pass:
            loss_value.backward()

            # update parameters
            model.update_parameters(learning_rate)

            # Print progress every batch:
            batch_loss = loss_value.get_plain_text()
            if rank == 0:
                pass
                print(f"\tBatch {(batch + 1)} of {num_batches} Loss {batch_loss.item():.4f}")

    # the execution environment keeps all printed strings in the printed variable
    return printed  # noqa: F821


print("[%] Starting computation")
result = run_encrypted_training()
print(result[0])
