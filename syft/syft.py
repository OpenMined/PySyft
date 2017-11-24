import numpy as np
import zmq
from random import choice

class FloatTensor():

    
    def __init__(self, controller, data, data_is_pointer = False, verbose=False):
        self.verbose = verbose
        self.controller = controller
        if(data is not None and not data_is_pointer):
            
            controller.socket.send_json({"functionCall":"createTensor", "data": list(data.flatten()), "shape": data.shape})
            
            self.id = int(controller.socket.recv_string())
        
            print("FloatTensor.__init__: " +  str(self.id))

        elif(data_is_pointer):
            self.id = int(data)


    def __add__(self,x):

        self.controller.socket.send_json(self.cmd("add",[x.id])) # sends the command
        return FloatTensor(self.controller,int(self.controller.socket.recv_string()),True)

    
    def abs(self):
        return self.no_params_func("abs")

    def add_(self, x):
        return self.params_func("add_",[x])

    def neg(self):
        return self.no_params_func("neg")

    def scalar_multiply(self, scalar):
        return self.params_func("scalar_multiply",[scalar])
    
    def params_func(self, name, params, return_response=False):
        
        # send the command
        self.controller.socket.send_json(self.cmd(name,tensorIndexParams=params))
        # receive output from command
        res = self.controller.socket.recv_string()

        if(self.verbose):
            print(res)

        if(return_response):
            return res
        return self

    def no_params_func(self, name, return_response=False):
        return( self.params_func(name,[],return_response) )
    
    def __repr__(self):
        return self.no_params_func("print",True)

    def __str__(self):
        return self.no_params_func("print",True)

    def cmd(self,functionCall,tensorIndexParams=[]):
        cmd = {}
        cmd['functionCall'] = functionCall
        cmd['objectType'] = 'tensor'
        cmd['objectIndex'] = self.id
        cmd['tensorIndexParams'] = tensorIndexParams
        return cmd


class SyftController():

    def __init__(self, identity):

        self.identity = identity

        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, identity)
        self.socket.connect("tcp://localhost:5555")

    def FloatTensor(self,data):
        return FloatTensor(self,data)

