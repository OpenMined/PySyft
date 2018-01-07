import zmq
import uuid
import syft.tensor
from syft.utils import DelayedKeyboardInterrupt

identity = str(uuid.uuid4())
context = zmq.Context()

socket = context.socket(zmq.DEALER)
socket.setsockopt_string(zmq.IDENTITY, identity)
socket.connect("tcp://localhost:5555")
verbose = False

class TensorPointer():
    def __init__(self):
        ""
    def __getitem__(self,id):
        return get_tensor(id)

class ModelPointer():
    def __init__(self):
        ""
    def __getitem__(self,id):
        return get_model(id)        

tensors = TensorPointer()        
models = ModelPointer()

def log(message):
    if (verbose):
        print(message)

# Network Convenience Functions
def cmd(functionCall, params=[]):
    cmd = {
        'functionCall': functionCall,
        'objectType': 'controller',
        'objectIndex': '-1',
        'tensorIndexParams': params}
    return cmd

# Introspection
def num_models():
    return no_params_func(cmd,"num_models",'int')

def get_model(id):
    return syft.nn.Model(id=int(id)).discover()

def load(filename):
    return params_func(cmd,"load_floattensor",params=[filename], return_type='FloatTensor')

def save(x,filename):
    return x.save(filename)

def num_tensors():
    return no_params_func(cmd,"num_tensors",'int')

def new_tensors_allowed(allowed=None):
    if(allowed is None):
        return no_params_func(cmd,"new_tensors_allowed",'bool')
    else:
        if(allowed):
            return params_func(cmd, "new_tensors_allowed",["True"], 'bool')
        else:
            return params_func(cmd, "new_tensors_allowed",["False"], 'bool')

def get_tensor(id):
    return syft.tensor.FloatTensor(data=int(id),data_is_pointer=True)

def __getitem__(id):
        return get_tensor(id)

def params_func(cmd_func, name, params, return_type=None):
    # makes things fail gracefully - without this, interruping the process early
    # can cause the client and the server to go out of sync
    with DelayedKeyboardInterrupt():
        # send the command
        socket.send_json(
            cmd_func(name, params=params))
        # receive output from command
        res = socket.recv_string()

        if("Unity Error:" == res[0:12]):
            raise Exception(res)

        if(verbose):
            print(res)

        if(return_type is None):
            return None
        elif(return_type == 'FloatTensor'):
            if(res != '-1' and res != ''):
                if(verbose):
                    print("FloatTensor.__init__: " +  res)
                return syft.tensor.FloatTensor(data=int(res),data_is_pointer=True)
            return None
        elif(return_type == 'IntTensor'):
            if(res != '-1' and res != ''):
                if(verbose):
                    print("IntTensor.__init__: " +  res)
                return syft.tensor.IntTensor(data=int(res),data_is_pointer=True)
            return None
        elif return_type == 'FloatTensor_list':
            if(res != ''):
                tensors = list()
                if(res[-1] == ','):
                    res = res[:-1]
                for str_id in res.split(","):
                    tensors.append(get_tensor(int(str_id)))
                return tensors
            else:
                return []
        elif return_type == "Model_list":
            models = list()
            if(res[-1] == ','):
                res = res[:-1]
            for str_id in res.split(","):
                models.append(get_model(int(str_id)))
            return models
        elif return_type == 'int':
            return int(res)
        elif return_type == 'string':
            return str(res)
        elif return_type == 'bool':
            if res == 'True':
                return True
            elif res == 'False':
                return False
            else:
                return res
        else:
            return res

def send_json(message,response=True):
    
        # send the command
        socket.send_json(message)

        # receive output from command
        res = socket.recv_string()

        if("Unity Error:" == res[0:12]):
            raise Exception(res)
        return res

def no_params_func(cmd_func, name, return_type):
    return params_func(cmd_func, name, [], return_type)

