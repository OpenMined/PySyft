import os
import json

from pathlib import Path

from . import utils


def torch2ipfs(model):
    pass


def ipfs2torch(model_addr):
    pass


def serialize_torch_model(model, **kwargs):
    """
    kwargs are the arguments needed to instantiate the model
    """
    state = {'state_dict': model.state_dict(), 'kwargs': kwargs}
    torch.save(state, 'temp_model.pth.tar')
    with open('temp_model.pth.tar', 'rb') as f:
        model_bin = f.read()
    return model_bin


def deserialize_torch_model(model_bin, model_class, **kwargs):
    """
    model_class is needed since PyTorch uses pickle for serialization
        see https://discuss.pytorch.org/t/loading-pytorch-model-without-a-code/12469/2 for details
    kwargs are the arguments needed to instantiate the model from model_class
    """
    with open('temp_model2.pth.tar', 'wb') as g:
        g.write(model_bin)
    state = torch.load()
    model = model_class(**state['kwargs'])
    model.load_state_dict(state['state_dict'])
    return model


def save_best_torch_model_for_task(task, model):
    utils.ensure_exists(f'{Path.home()}/.openmined/models.json', {})
    with open(f"{Path.home()}/.openmined/models.json", "r") as model_file:
        models = json.loads(model_file.read())

    models[task] = torch2ipfs(model)

    with open(f"{Path.home()}/.openmined/models.json", "w") as model_file:
        json.dump(models, model_file)


def best_torch_model_for_task(task, return_model=False):
    if not os.path.exists(f'{Path.home()}/.openmined/models.json'):
        return None

    with open(f'{Path.home()}/.openmined/models.json', 'r') as model_file:
        models = json.loads(model_file.read())
        if task in models.keys():
            if return_model:
                return ipfs2torch(models[task])
            else:
                return models[task]

    return None
