import requests
import json
import keras
import os
import numpy as np
import time

import grid.bygone as by

## Base IPFS Address
IPFS_ADDR = 'https://ipfs.infura.io/ipfs'

class Server():

    def poll(self, interval):
        while True:
            job = by.get_job()
            if job == None:
                print('no jobs, tryin again in seconds'.format(interval))
                time.sleep(interval)
            else:
                model = self.run_experiment(job)
                self.save_experiment(job, model)

    def save_experiment(self, job, model):
        model_file = 'trained-model.h5'
        model.save(model_file)
        model_config = {
            'file': ('model', open(model_file, 'rb'), 'application/octet-stream'),
        }
        r = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=model_config)
        model_response = json.loads(r.text)

        by.add_result(job, model_response["Hash"])
        os.remove(model_file)

    def run_experiment(self, ipfs_address):
        r = requests.get('{}/{}'.format(IPFS_ADDR, ipfs_address))
        json_response = json.loads(r.text)

        print('got resp {}'.format(json_response))

        model = self.load_model(json_response["model"])
        input = self.load_tensor(json_response["input"])
        target = self.load_tensor(json_response["target"])

        batch_size = json_response["batch_size"]
        epochs = json_response["epochs"]

        model.fit(input, target, epochs=epochs, batch_size=batch_size, verbose=1)
        return model


    def load_model(self, ipfs_address):
        r = requests.get('{}/{}'.format(IPFS_ADDR, ipfs_address))

        # keras only loads models from files
        with open('job-model.h5', 'wb') as model_file:
            model_file.write(r.content)

        model = keras.models.load_model('job-model.h5')
        os.remove('job-model.h5')

        return model

    def load_tensor(self, ipfs_address):
        r = requests.get('{}/{}'.format(IPFS_ADDR, ipfs_address))

        with open('job-tensor.npy', 'wb') as tensor_file:
            tensor_file.write(r.content)

        tensor = np.load('job-tensor.npy')
        os.remove('job-tensor.npy')

        return tensor
