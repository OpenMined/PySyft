#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Arguments:
    epochs = 1
    
    n_train_items = 128*4
    n_test_items = 128*4
    
    batch_size = 128
    test_batch_size = 128
    
    protocol = "fss"
    precision_fractional = 3
    lr = 0.1
    log_interval = 40

args = Arguments()


# # Model A
# 
# 
# Layer Input Size Description Output
# 
# Fully Connected Layer 28×28 Fully connected layer 128
# 
# ReLU Activation 128 ReLU(·) on each input 128
# 
# Fully Connected Layer 128 Fully connected layer 128
# 
# ReLU Activation 128 ReLU(·) on each input 128
# 
# Fully Connected Layer 128 Fully connected layer 10
# 
# ReLU Activation 10 ReLU(·) on each input 1
# 
# 
# Figure 3: Neural Network architecture from SecureML [6] for training over
# MNIST dataset.
# 
# 

# In[2]:


import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import syft as sy

from examples.experimental.FSS.utils import train, test, save, load
from examples.experimental.FSS.utils import get_private_data_loaders, get_public_data_loaders, estimate_time

hook = sy.TorchHook(torch) 
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

workers = [alice, bob]
sy.local_worker.clients = workers

kwargs = dict(crypto_provider=crypto_provider, protocol=args.protocol, requires_grad=True)

private_train_loader, private_test_loader = get_private_data_loaders(workers, args, kwargs)
public_train_loader, public_test_loader = get_public_data_loaders(workers, args, kwargs)


# In[3]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x


# In[4]:


public_model = Net()
name = "model-A-public"

try:
    load(public_model, name)
    print('Loaded!')
except FileNotFoundError:
    optimizer = optim.SGD(public_model.parameters(), lr=args.lr)

    accuracies = []
    for epoch in range(1, args.epochs + 1):    
        train_time = train(args, public_model, public_train_loader, optimizer, epoch)
        test_time, acc = test(args, public_model, public_test_loader)
        accuracies.append(acc)

    print(accuracies)
    save(public_model, name)


# In[5]:


n_instances = (550 * args.n_test_items) * args.epochs
print(n_instances)
sy.local_worker.crypto_store.provide_primitives(
    ["fss_comp"], [alice, bob], n_instances=n_instances
)

sy.local_worker.crypto_store.provide_primitives(
    ["fss_eq"], [alice, bob], n_instances=2*args.n_test_items
)


BS = args.batch_size
n_instances = 2* int(2 * args.n_train_items / BS)
sy.local_worker.crypto_store.provide_primitives(
    ["beaver"], [alice, bob], n_instances=n_instances, beaver={'op_shapes': [
        ('matmul', (BS, 784), (784, 128)),
        ('matmul', (BS, 128), (128, 128)),
        ('matmul', (BS, 128), (128, 10)),
        ('matmul', (BS, 10), (10, 128)),
        ('matmul', (128, BS), (BS, 10)),
        ('matmul', (BS, 128), (128, 128)),
        ('matmul', (128, BS), (BS, 128)),
        ("matmul", (BS, 128), (128, 784)),
        ("matmul", (784, BS), (BS, 128)),
        ('mul', (BS, 128), (BS, 128)),
        ('mul', (BS, 10), (BS, 10)),
    ]}
)
for i in range(n_instances):
    sy.local_worker.crypto_store.provide_primitives(
        ["beaver"], [alice, bob], n_instances=1, beaver={'op_shapes': [
            ('mul', (BS, 10), (1, )),
        ]}
    )
sy.local_worker.crypto_store.provide_primitives(
    ["beaver"], [alice, bob], n_instances=5*n_instances, beaver={'op_shapes': [
        ('mul', (BS, 10), (BS, 10)),
        ('mul', (BS, 128), (BS, 128)),
    ]}
)


# In[6]:


model = public_model#Net()
model.fix_precision(precision_fractional=args.precision_fractional).share(*workers, **kwargs)

t = time.time()
test_time, acc = test(args, model, private_test_loader)
print(time.time() - t)


# In[ ]:


assert False


# In[5]:


if True:
    # 550 * args.n_train_items  + 
    n_instances = (550 * args.n_test_items) * args.epochs
    print(n_instances)
    sy.local_worker.crypto_store.provide_primitives(
        ["fss_comp"], [alice, bob], n_instances=n_instances
    )

    sy.local_worker.crypto_store.provide_primitives(
        ["fss_eq"], [alice, bob], n_instances=2*args.n_test_items
    )


# In[6]:


if True:
    BS = args.batch_size
    n_instances = 2* int(2 * args.n_train_items / BS)
    sy.local_worker.crypto_store.provide_primitives(
        ["beaver"], [alice, bob], n_instances=n_instances, beaver={'op_shapes': [
            ('matmul', (BS, 784), (784, 128)),
            ('matmul', (BS, 128), (128, 128)),
            ('matmul', (BS, 128), (128, 10)),
            ('matmul', (BS, 10), (10, 128)),
            ('matmul', (128, BS), (BS, 10)),
            ('matmul', (BS, 128), (128, 128)),
            ('matmul', (128, BS), (BS, 128)),
            ("matmul", (BS, 128), (128, 784)),
            ("matmul", (784, BS), (BS, 128)),
            ('mul', (BS, 128), (BS, 128)),
            ('mul', (BS, 10), (BS, 10)),
        ]}
    )
    for i in range(n_instances):
        sy.local_worker.crypto_store.provide_primitives(
            ["beaver"], [alice, bob], n_instances=1, beaver={'op_shapes': [
                ('mul', (BS, 10), (1, )),
            ]}
        )
    sy.local_worker.crypto_store.provide_primitives(
        ["beaver"], [alice, bob], n_instances=5*n_instances, beaver={'op_shapes': [
            ('mul', (BS, 10), (BS, 10)),
            ('mul', (BS, 128), (BS, 128)),
        ]}
    )


# In[8]:


model = public_model#Net()
public_model.fix_precision(precision_fractional=args.precision_fractional).share(*workers, **kwargs)

optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision(precision_fractional=args.precision_fractional) 

test_time, acc = test(args, model, private_test_loader)

accuracies = []
for epoch in range(1, args.epochs + 1):    
    train_time = train(args, model, private_train_loader, optimizer, epoch)
    test_time, acc = test(args, model, private_test_loader)
    accuracies.append(acc)
print(accuracies)


# In[ ]:


estimate_time(train_time, test_time, args.batch_size, 15, 50_000, 10_000)


# ### Info

# In[ ]:


# Influence of the number of bits of encrypting a pretrained network
precfrac_bits_acc = {
    1: {
        32: 9.50
    }
    2: {
        32: 98.05,
        28: 98.04,
        24: 98.05,
        20: 97.90,
        16: 95.97,
        12: 69.44,
    },
    3: {
        32: 98.1,
        28: 98.17,
        24: 98.08,
        20: 96.94,
        16: 76.20,
        12:, 10.35,
    }
}


# In[ ]:


# Plain text acc vs encrypted acc on training n_epoch=15
plaintext = [
    [94.4, 96.0, 96.7, 97.0, 97.4, 97.5, 97.7, 97.9, 97.9, 98.0, 98.0, 98.0, 98.1, 98.1, 98.1],
    [94.2, 96.0, 96.9, 97.2, 97.4, 97.6, 97.7, 97.7, 97.8, 97.8, 97.8, 97.9, 97.9, 98.0, 98.0]
]
encrypted = [
    [85.2, 86.6, 87.1, 96.7, 96.9, 97.2, 97.4, 97.5, 97.5]
]

n_train_items = -1
n_test_items = -1
batch_size = 128
test_batch_size = 128
precision_fractional = 3
lr = 0.1


# In[ ]:


# Test + Retraining encrypted from encrypted net
trained_on = [94.2, 96.0, 96.9, 97.2, 97.4, 97.6, 97.7, 97.7, 97.8, 97.8, 97.8, 97.9, 97.9, 98.0, 98.0]
encrypted = [
    [98.1*, 97.8, 98.0, 97.9]
]

n_train_items = -1
n_test_items = -1
batch_size = 128
test_batch_size = 128
precision_fractional = 3
lr = 0.1


# In[ ]:


# Plain text acc vs encrypted acc on training n_epoch=5
plaintext = [
    [75, 84, 88, 89, 89],
    [62, 76, 78, 79, 80],
]
encrypted = [
    [65, 76, 78, 81, 81],
    [61, 79, 81, 81, 82]
]

n_train_items = 128*32
n_test_items = 128*32
batch_size = 128
test_batch_size = 128
precision_fractional = 3
lr = 0.1


# In[ ]:




