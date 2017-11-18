import numpy as np
import zmq
from random import choice

class FloatTensor():
    
    def __init__(self, controller, data, shape):
        
        self.controller = controller
        self.data = data
        self.shape = shape
        
        controller.socket.send_json({"functionCall":"createTensor", "data": data, "shape": shape})
        
        self.id = int(controller.socket.recv_string())
        
        print("FloatTensor.__init__: " +  str(self.id))
    
    def init_sigmoid_matrix_multiply(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"init_sigmoid_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        return self.controller.socket.recv_string()
    
    def init_add_matrix_multiply(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"init_add_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        return self.controller.socket.recv_string()
    
    def init_weights(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"init_weights","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        return self.controller.socket.recv_string()
    
    def sigmoid_matrix_multiply(self, tensor_1, tensor_2):
        
        self.controller.socket.send_json({"functionCall":"sigmoid_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id, tensor_2.id]})
        return self.controller.socket.recv_string()
    
    def reset_weights(self):
        
        self.controller.socket.send_json({"functionCall":"reset_weights","objectType":"tensor","objectIndex":self.id})
        return self.controller.socket.recv_string()
    
    def inline_elementwise_subtract(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"inline_elementwise_subtract","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        return self.controller.socket.recv_string()
    
    def multiply_derivative(self, tensor_1):
        
        self.controller.socket.send_json({"functionCall":"multiply_derivative","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id]})
        return self.controller.socket.recv_string()
    
    def add_matrix_multiply(self, tensor_1, tensor_2):
        
        self.controller.socket.send_json({"functionCall":"add_matrix_multiply","objectType":"tensor","objectIndex":self.id,"tensorIndexParams":[tensor_1.id, tensor_2.id]})
        return self.controller.socket.recv_string()
    
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
    
    def FloatTensor(self,data,shape):
        return FloatTensor(self,data,shape)
