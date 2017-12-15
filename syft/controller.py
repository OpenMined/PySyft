import zmq
import uuid
import syft.tensor

identity = str(uuid.uuid4())
context = zmq.Context()

socket = context.socket(zmq.DEALER)
socket.setsockopt_string(zmq.IDENTITY, identity)
socket.connect("tcp://localhost:5555")
verbose = False

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

def num_tensors():
    return no_params_func(cmd,"num_tensors",'int')

def get_tensor(id):
    return syft.tensor.FloatTensor(controller=self,data=int(id),data_is_pointer=True)


def params_func(cmd_func, name, params, return_type=None):
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
            if(verbose):
                print("FloatTensor.__init__: " +  res)
            return syft.tensor.FloatTensor(data=int(res),data_is_pointer=True)
        elif return_type == 'FloatTensor_list':
            tensors = list()
            if(res[-1] == ','):
                res = res[:-1]
            for str_id in res.split(","):
                tensors.append(syft.tensor.FloatTensor(data=int(str_id),data_is_pointer=True))
            return tensors
        elif return_type == 'int':
            return int(res)
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

