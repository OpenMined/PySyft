from grid import ipfsapi
import base64
import random
import keras
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from grid.pubsub.worker import Worker

gridWorker = Worker()
gridWorker.work()

print("grid id: {}".format(gridWorker.id))
