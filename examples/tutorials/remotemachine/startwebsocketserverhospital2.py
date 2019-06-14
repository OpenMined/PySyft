# remote.py
import syft as sy
import sys
import torch
import asyncio
from sklearn.preprocessing import StandardScaler

hook = sy.TorchHook(torch, verbose=True)
torch.manual_seed(1)

configs = {"id": "hospital_2", "host": "localhost", "hook": hook, "verbose": False, "port": 8084}


async def show_all(worker):
    await asyncio.sleep(0)
    while True:
        print("Objects:", worker._objects)
        await asyncio.sleep(2.0)


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
# Standardise data for input into Neural Net
scaler = StandardScaler()
transformed = scaler.fit_transform(data.data[int(len(data.data) / 2) :])
data_features = torch.tensor(transformed, requires_grad=True).tag("#train_features")
data_target = torch.tensor(data.target[int(len(data.data) / 2) :]).tag("train_target")


worker = sy.workers.WebsocketServerWorker(data=[data_features, data_target], **configs)

# loop = asyncio.get_event_loop()
# loop.create_task(show_all(worker))

worker.start()
