#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch import optim

import syft as sy
hook = sy.TorchHook(torch)


# create a couple workers

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")



data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

# get pointers to training data on each worker by
# sending some training data to bob and alice
data_bob = data[0:2]
target_bob = target[0:2]

data_alice = data[2:]
target_alice = target[2:]

# Iniitalize A Toy Model
model = nn.Linear(2,1)

data_bob = data_bob.send(bob)
data_alice = data_alice.send(alice)
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice)

# organize pointers into a list
datasets = [(data_bob,target_bob),(data_alice,target_alice)]

"""
changes are done on line : 45,46,48 and 52
"""

opt_bob= optim.Adagrad(params=model.parameters(),lr=0.1)
opt_alice = optim.SGD(params=model.parameters(),lr=0.1)

optimizer = [opt_bob, opt_alice]

def train():
    # Training Logic
    optimizer = [opt_bob, opt_alice]
  
    for iter in range(10):
        
        # NEW) iterate through each worker's dataset
        for data,target in datasets:
            
            # NEW) send model to correct worker
            model.send(data.location)

            # 1) erase previous gradients (if they exist)
            opt.zero_grad()

            # 2) make a prediction
            pred = model(data)

            # 3) calculate how much we missed
            loss = ((pred - target)**2).sum()

            # 4) figure out which weights caused us to miss
            loss.backward()

            # 5) change those weights
            opt.step()
            
            # NEW) get model (with gradients)
            model.get()

            # 6) print our progress
            print(loss.get()) # NEW) slight edit... need to call .get() on loss\
    

train()



