import json
import datetime
import os
import requests
import numpy as np
import uuid
from colorama import Fore, Back, Style
import ipywidgets as widget
import threading
import time
from tempfile import TemporaryFile

class Grid():

    def __init__(self):
        self.jobId = None

    def configuration(self, model, epochs, batch_size):
        configuration = GridConfiguration(model, epochs, batch_size)
        return configuration

    def fit(self, input, target, configurations, name=None):

        # save input / target arrays

        np.save('tmp-input', input)
        np.save('tmp-target', target)

        input_config = { 'file': ('input', open('tmp-input.npy', 'rb'), 'application/octet-stram')}
        target_config = { 'file': ('target', open('tmp-target.npy', 'rb'), 'application/octet-stream')}

        input_r = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=input_config)
        target_r = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=target_config)

        all_jobs = []
        for i in range(0, len(configurations)):
            config = configurations[i]

            # Start by saving the model
            model_file = f'tmp-{i}.h5'
            model = config.model.save(model_file)
            model_config = {
                'file': ('model', open(model_file, 'rb'), 'application/octet-stream'),
            }
            r = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=model_config)
            model_response = json.loads(r.text)

            # Now save the models config
            job_config_json = {
                'model': model_response["Hash"],
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'input': json.loads(input_r.text)["Hash"],
                'target': json.loads(target_r.text)["Hash"]
            }

            job_config = {
                'file': ('job_config', json.dumps(job_config_json), 'application/octet-stream')
            }

            r = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=job_config)
            all_jobs.append(json.loads(r.text)["Hash"])


        experiment_config_json = {
            'jobs': all_jobs
        }

        experiment_config = {
            'file': ('experiment', json.dumps(experiment_config_json), 'application/octet-stream')
        }

        experiment_r = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=experiment_config)
        print(experiment_r.text)

        os.remove('tmp-input.npy')
        os.remove('tmp-target.npy')

            # all_models.append(j["Hash"])
            # print(r.request.body)

            # https://ipfs.infura.io:5001/api/v0/add?stream-channels=true


    def check_experiment_status(self, experiments, status_widgets):
        for i in range(0, len(experiments)):
            experiment = experiments[i]
            widget = status_widgets[i]

            widget.value = self.controller.send_json({
                "objectType": "Grid",
                "functionCall": "checkStatus",
                "experimentId": experiment["jobId"]
            })

    def get_experiments(self):
        if not os.path.exists(".openmined/grid/experiments.json"):
            print(f'{Back.RED}{Fore.WHITE} No experiments found {Style.RESET_ALL}')
            return

        names = []
        uuids = []
        status = []

        with open('.openmined/grid/experiments.json', 'r') as outfile:
            d = json.loads(outfile.read())
            print(f"{Back.BLACK}{Fore.WHITE} ALL EXPERIMENTS {Style.RESET_ALL}")
            for experiment in d:
                names.append(widget.Label(experiment["name"]))
                uuids.append(widget.Label(experiment["uuid"]))
                status.append(widget.Label("Checking..."))

            names_column = widget.VBox(names)
            uuid_column = widget.VBox(uuids)
            status_column = widget.VBox(status)

            check_status_thread = threading.Thread(target=self.check_experiment_status, args=(d, status))
            check_status_thread.start()

            box = widget.HBox([names_column, uuid_column, status_column])
            box.border = '10'
            return box

    def store_job(self, jobId, name=None):
        if name is None:
            now = datetime.datetime.now()
            name = 'Experiment on {}-{}-{}'.format(now.day, now.month, now.year)

        if not os.path.exists(".openmined/grid"):
            os.makedirs(".openmined/grid")

        if not os.path.exists(".openmined/grid/experiments.json"):
            with open('.openmined/grid/experiments.json', 'w') as outfile:
                json.dump([], outfile)

        d = None
        with open('.openmined/grid/experiments.json', 'r') as outfile:
            d = json.loads(outfile.read())

        with open('.openmined/grid/experiments.json', 'w') as outfile:
            newExperiment = {
                "name": name,
                "jobId": jobId,
                "uuid": str(uuid.uuid4())
            }
            # most recent first
            d.insert(0, newExperiment)
            json.dump(d, outfile)

    def get_results(self, experiment=None):
        if not os.path.exists(".openmined/grid/experiments.json") and self.jobId is None:
            raise Exception("There are no saved experiments and you have not submitted a job.")

        if not os.path.exists(".openmined/grid/experiments.json") and experiment is None:
            raise Exception("There are no saved experiments.  Submit a job first.")

        usedJob = None
        if not experiment is None:
            if not os.path.exists(".openmined/grid/experiments.json"):
                raise Exception("There are no saved experiments.")
            with open('.openmined/grid/experiments.json', 'r') as outfile:
                d = json.loads(outfile.read())
                for __experiment in d:
                    if experiment == __experiment["uuid"]:
                        usedJob = __experiment["jobId"]

        if usedJob is None and not experiment is None:
            raise Exception(f"No experiments matching {Fore.GREEN}{experiment}{Style.RESET_ALL}")

        if usedJob is None and not self.jobId is None:
            usedJob = self.jobId

        if usedJob is None:
            raise Exception("There are no saved experiments and you have not submitted a job.")

        results = self.controller.send_json({
            "objectType": "Grid",
            "functionCall": "getResults",
            "experimentId": usedJob
        })

        modelIds = json.loads(results)
        return ExperimentResults(list(map(lambda id: nn.Sequential(id=id), modelIds)))

class ExperimentResults():
    def __init__(self, models):
        self.results = models

class GridConfiguration():
    def __init__(self, model, epochs, batch_size):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def toJSON(self):
        return {
            "model": self.model.id,
            "lr": self.lr,
            "criterion": self.criterion,
            "iters": self.iters
        }
