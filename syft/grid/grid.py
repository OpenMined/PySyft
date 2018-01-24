import syft.controller
import syft.nn as nn
import json
import datetime
import os

import uuid
from colorama import Fore, Back, Style

class Grid():

    def __init__(self):
        self.controller = syft.controller
        self.jobId = None

    def configuration(self, model, lr, criterion, iters):
        configuration = GridConfiguration(model, lr, criterion, iters)
        return configuration

    def learn(self, input, target, configurations, name=None):
        configurations_json = list(map(lambda x: x.toJSON(), configurations))
        self.jobId = self.controller.send_json({"objectType": "Grid",
                                   "functionCall": "learn",
                                   "tensorIndexParams": [input.id, target.id],
                                   "configurations": configurations_json})

        self.store_job(self.jobId, name)

    def get_experiments(self):
        if not os.path.exists(".openmined/grid/experiments.json"):
            print(f'{Back.RED}{Fore.WHITE} No experiments found {Style.RESET_ALL}')
            return

        with open('.openmined/grid/experiments.json', 'r') as outfile:
            d = json.loads(outfile.read())
            print(f"{Back.BLACK}{Fore.WHITE} ALL EXPERIMENTS {Style.RESET_ALL}")
            print(f"Get the result of your experiment by calling {Fore.GREEN}get_results{Style.RESET_ALL} with the highlighted uuid")
            print("")
            for experiment in d:
                name = experiment["name"]
                uuid = experiment["uuid"]

                print(f"    - {name} ({Fore.GREEN}{uuid}{Style.RESET_ALL})")

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
    def __init__(self, model, lr, criterion, iters, name=None):
        self.model = model
        self.lr = lr
        self.criterion = criterion
        self.iters = iters

    def toJSON(self):
        return {
            "model": self.model.id,
            "lr": self.lr,
            "criterion": self.criterion,
            "iters": self.iters
        }
