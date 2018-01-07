import syft.controller


class Grid():

    def __init__(self):
        self.controller = syft.controller

    def configuration(self, model, lr):
        configuration = GridConfiguration(model, lr)
        return configuration

    def learn(self, input, target, configurations):
        configurations_json = list(map(lambda x: x.toJSON(), configurations))
        self.controller.send_json({"objectType": "Grid",
                                   "functionCall": "learn",
                                   "tensorIndexParams": [input.id, target.id],
                                   "configurations": configurations_json})


class GridConfiguration():
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def toJSON(self):
        return {"model": self.model.id,
                "lr": self.lr}
