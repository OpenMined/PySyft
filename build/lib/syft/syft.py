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


        command = {}
        command["functionCall"] = "add"
        command["objectType"] = "tensor"
        command["objectIndex"] = self.id
        command["tensorIndexParams"] = [x.id]

        self.controller.socket.send_json(command) # sends the command
        return FloatTensor(self.controller,int(self.controller.socket.recv_string()),True)

        return "Tensors don't have the same shape"

    
    def abs(self):
        command = {}
        command["functionCall"] = "abs"
        command["objectType"] = "tensor"
        command["objectIndex"] = self.id
        command["tensorIndexParams"] = []
        
        self.controller.socket.send_json(command) # sends the command
        res = self.controller.socket.recv_string() # receives output from command

        if(self.verbose):
            print(res)

        return self

    def neg(self):

        command = {}
        command["functionCall"] = "neg"
        command["objectType"] = "tensor"
        command["objectIndex"] = self.id
        command["tensorIndexParams"] = []

        self.controller.socket.send_json(command) # sends the command
        return self.controller.socket.recv_string() # receives output from command
    
    def init_sigmoid_matrix_multiply(self, tensor_1):
        command = {}
        command["functionCall"] = "init_sigmoid_matrix_multiply"
        command["objectType"] = "tensor"
        command["objectIndex"] = self.id
        command["tensorIndexParams"] = [tensor_1.id]
        
        self.controller.socket.send_json(command)
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def init_add_matrix_multiply(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"init_add_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def init_weights(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"init_weights","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def sigmoid_matrix_multiply(self, tensor_1, tensor_2):
        
        self.controller.socket.send_json({"functionCall":"sigmoid_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id, tensor_2.id]})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def reset_weights(self):
        
        self.controller.socket.send_json({"functionCall":"reset_weights","objectType":"tensor","objectIndex":self.id})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def inline_elementwise_subtract(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"inline_elementwise_subtract","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def multiply_derivative(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"multiply_derivative","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def add_matrix_multiply(self, tensor_1, tensor_2):
        
        self.controller.socket.send_json({"functionCall":"add_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id, tensor_2.id]})
        res = self.controller.socket.recv_string() # receives output from command
        
        if(self.verbose):
            print(res)

        return self
    
    def __repr__(self):
        self.controller.socket.send_json({"functionCall":"print","objectType":"tensor","objectIndex":self.id})
        res = self.controller.socket.recv_string() # receives output from command
        
        return res

    def __str__(self):
        self.controller.socket.send_json({"functionCall":"print","objectType":"tensor","objectIndex":self.id})
        res = self.controller.socket.recv_string() # receives output from command

        return res

    def print(self):
        self.controller.socket.send_json({"functionCall":"print","objectType":"tensor","objectIndex":self.id})
        
        print(self.controller.socket.recv_string())
    
class SyftController():
    
    def __init__(self, identity):
        
        self.identity = identity
        
        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, identity)
        self.socket.connect("tcp://localhost:5555")
    
    def FloatTensor(self,data):
        return FloatTensor(self,data)
