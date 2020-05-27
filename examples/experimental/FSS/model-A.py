#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Arguments:
    epochs = 15
    
    n_train_items = -1#128*32
    n_test_items = -1#128*32
    
    batch_size = 128
    test_batch_size = 128
    
    protocol = "fss"
    precision_fractional = 1
    lr = 0.1
    log_interval = 40

args = Arguments()


# # FALCON - Model A
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


model = public_model#Net()
model.fix_precision(precision_fractional=args.precision_fractional).share(*workers, **kwargs)

t = time.time()
test_time, acc = test(args, model, private_test_loader)
print(time.time() -t)


# In[6]:

