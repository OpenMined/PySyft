import syft.controller
import onnx

def export_as_json(model, input_tensor, verbose=False):
    return syft.controller.params_func(
        syft.controller.cmd, 
        "to_proto", 
        [model.id, input_tensor.id, "", "json"], 
        return_type='string'
    )

def export(model, input_tensor, filename, verbose=False):
    return syft.controller.params_func(
        syft.controller.cmd, 
        "to_proto", 
        [model.id, input_tensor.id, filename, "file"], 
        return_type='string'
    )

def load(filename):
    return syft.controller.params_func(
        syft.controller.cmd,
        "from_proto",
        [filename],
        return_type='Model'
    )
