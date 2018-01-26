# import json
# import yaml
# from .. import models as keras_models
# from .. import layers as keras_layers

from .. import models as keras_models
from .. import layers as keras_layers
import syft.controller as controller
import syft.nn as nn
import re

def model_from_json(json_string, custom_objects=None):
    # Preprocessing
    json_string = re.sub('Dense', 'Linear', json_string)
    json_string = re.sub('dense_', 'linear_', json_string)

    model_id = controller.send_json(controller.cmd("model_from_json", params=[json_string]))
    syft_model = nn.Model(model_id).discover()
    if syft_model._layer_type == "sequential":
        syft_model.layers = syft_model.models()

        model = keras_models.Sequential()
        model.syft = syft_model
        should_continue = False
        for i, layer in enumerate(model.syft.layers):
            if should_continue:
                should_continue = False
                continue

            if layer._layer_type == "linear":
                dense = keras_layers.Dense(layer.output_shape)
                dense.input_shape = (layer.input_shape, )
                dense.ordered_syft.append(layer)
                if i+1 < len(model.syft.layers):
                    if model.syft.layers[i+1]._layer_type in ["softmax", "relu", "logsoftmax"]:
                        dense.activation_str = model.syft.layers[i+1]._layer_type
                        dense.syft_activation = model.syft.layers[i+1]
                        dense.ordered_syft.append(model.syft.layers[i+1])

                        should_continue = True

                model.layers.append(dense)
        
    
    return model
    # config = json.loads(json_string)

    # if type(config) == dict:
    #     if config["class_name"] == "Sequential":
    #         model = keras_models.Sequential()
    #         layers = config["config"]
    #     else:
    #         raise Exception('Model %s is not implemented' % config["class_name"])
    # elif type(config) == list:
    #     model = Sequential()
    #     layers = config    
    # else:
    #     raise Exception('The JSON string must represent either an object or an array')

    # for i, layer in enumerate(layers):
    #     print(i, layer)
    #     if layer["class_name"] == "Dense":
    #         layer_config = layer["config"]

    #         if "batch_input_shape" in layer_config:
    #             input_shape = tuple(layer_config["batch_input_shape"][1:])
    #         else:
    #             if i == 0:
    #                 raise Exception("The batch_input_shape property must be set for the first layer of a Sequential model")
    #             else:
    #                 input_shape = model.layers[-1].output_shape

    #         if "activation" in layer_config:
    #             activation_str = layer_config["activation"]
    #         else:
    #             activation_str = None

    #         model.add(keras_layers.Dense(layer_config["units"], input_shape, activation_str))
    #         layers = config["config"]
    #     else:
    #         raise Exception('Layer %s is not implemented' % config["class_name"])
    
    # return model