import pandas as pd
import json


class Worker():
    """Need to a way send the worker class object
    #TODO remove pandas functions from here and add to the pandas hook"""
    """
    Example
    adam = Worker("Adam")
    adam.LocalData('acme.csv')
    adam.AppendToLocalData('acme.csv')
    trask = Worker("Trask")
    trask.WorkerRemove()
    """
    workers = []
    id_list = []
    instance = []
    Name_list = []
    id_ = 0

    def __init__(self, name):
        self.name = name
        self.id_ = Worker.id_
        if self.name in Worker.Name_list:
            print("id already exists")
        else:
            Worker.workers.append({self.id_: self.name})
            Worker.id_list.append(self.id_)
            Worker.instance.append(self)
            Worker.Name_list.append(self.name)
            Worker.id_ = Worker.id_ + 1

    def WorkerRemove(self):
        if self.id_ not in Worker.id_list:
            return "id does not exist with same name"
        else:
            for i in Worker.workers:
                if [self.id_] == list(i.keys()):
                    Worker.workers.remove(i)
                    Worker.id_list.remove(self.id_)

    def Parameters(self):
        LocalData = {'Local DataFrame': self.frame}
        name = json.dumps({'name': self.name})
        return (LocalData, name, self)

    def DataFrameShape(self):
        return self.frame.shape

    def LocalData(self, path_to_data):
        self.frame = pd.read_csv(path_to_data)

    def AppendToLocalData(self, path_to_csv):
        self.frame = self.frame.append(pd.read_csv(path_to_csv))

    def concat(self, pandaframe, axis):
        self.frame()


class PandasHook():
    """#TODO Hook the pandas functions here and remove all the pandas
    functions from the Workers class, so that you could add more to the
    pandas functions"""
    def __init__(self):
        pass
