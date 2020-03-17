import torch
import syft
from syft.frameworks.crypten.context import toy_func, run_party
import crypten.communicator as comm
import crypten


def load(tag: str, src: int, id_worker: int):
    if comm.get().get_rank() == src:
        worker = syft.local_worker.get_worker(id_worker)
        result = worker.search(tag)[0].get()

        # file contains torch.tensor
        if torch.is_tensor(result):
            # Broadcast load type
            load_type = torch.tensor(0, dtype=torch.long)
            comm.get().broadcast(load_type, src=src)

            # Broadcast size to other parties.
            dim = torch.tensor(result.dim(), dtype=torch.long)
            size = torch.tensor(result.size(), dtype=torch.long)

            comm.get().broadcast(dim, src=src)
            comm.get().broadcast(size, src=src)
            result = crypten.mpc.MPCTensor(result, src=src)
    else:
        # Receive load type from source party
        load_type = torch.tensor(-1, dtype=torch.long)
        comm.get().broadcast(load_type, src=src)

        # Load in tensor
        if load_type.item() == 0:
            # Receive size from source party
            dim = torch.empty(size=(), dtype=torch.long)
            comm.get().broadcast(dim, src=src)
            size = torch.empty(size=(dim.item(),), dtype=torch.long)
            comm.get().broadcast(size, src=src)
            result = crypten.mpc.MPCTensor(torch.empty(size=tuple(size.tolist())), src=src)
        else:
            raise TypeError("Unrecognized load type on src")
    # TODO: Encrypt modules before returning them

    return result


def crypten_to_torch_modules(model):
    """Converts crypten modules to torch ones."""

    for name, curr_module in model._modules.items():
        # module_name = curr_module.__class__.split('.')[-1]
        if isinstance(curr_module, crypten.nn.module.Linear):
            out_nodes, in_nodes = curr_module._parameters['weight'].size()
            new_module = torch.nn.Linear(in_nodes, out_nodes)
            # Copy weights and biases
            weights = curr_module._parameters['weight']
            biases = curr_module._parameters['bias']
            new_module._parameters['weight'] = weights.get_plain_text()
            new_module._parameters['bias'] = biases.get_plain_text()
            model._modules[name] = new_module
        elif isinstance(curr_module, crypten.nn.module.ReLU):
            model._modules[name] = torch.nn.ReLU()

    return model


__all__ = ["toy_func", "run_party", "load"]
