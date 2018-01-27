import requests
import json
import keras
import os
import numpy as np

## Base IPFS Address
IPFS_ADDR = 'https://ipfs.infura.io/ipfs'

class Server():

    def run_experiment(self, ipfs_address):
        r = requests.get(f'{IPFS_ADDR}/{ipfs_address}')
        json_response = json.loads(r.text)

        print(f'got resp {json_response}')

        model = self.load_model(json_response["model"])
        input = self.load_tensor(json_response["input"])
        target = self.load_tensor(json_response["target"])

        batch_size = json_response["batch_size"]
        epochs = json_response["epochs"]


    def load_model(self, ipfs_address):
        r = requests.get(f'{IPFS_ADDR}/{ipfs_address}')

        # keras only loads models from files
        with open('job-model.h5', 'wb') as model_file:
            model_file.write(r.content)

        model = keras.models.load_model('job-model.h5')
        os.remove('job-model.h5')

        return model

    def load_tensor(self, ipfs_address):
        r = requests.get(f'{IPFS_ADDR}/{ipfs_address}')

        with open('job-tensor.npy', 'wb') as tensor_file:
            tensor_file.write(r.content)

        tensor = np.load('job-tensor.npy')
        os.remove('job-tensor.npy')

        return tensor
