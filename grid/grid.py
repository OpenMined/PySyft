from . import ipfsapi
import base64
import random
import keras
import json
import numpy as np 


class Grid(object):
    
    def __init__(self,ipfs_addr='127.0.0.1',port=5001):
        
        self.api = ipfsapi.connect(ipfs_addr, port)
        self.encoded_id = self.get_encoded_id()
        self.id = self.api.config_show()['Identity']['PeerID']
        
    def get_encoded_id(self):
        
        """Currently a workaround because we can't figure out how to decode the 'from' 
        side of messages sent across the wire. However, we can check to see if two messages
        are equal. Thus, by sending a random message to ourselves we can figure out what
        our own encoded id is. TODO: figure out how to decode it."""
        
        rand_channel = random.randint(0,1000000)
        temp_channel = self.api.pubsub_sub(topic=rand_channel,stream=True)
        secret = random.randint(0,1000000)
        self.api.pubsub_pub(topic=rand_channel,payload="id:" + str(secret))
        
        for encoded in temp_channel:

            # decode message
            decoded = self.decode_message(encoded)
            
            if(decoded is not None):
                if(str(decoded['data'].split(":")[-1]) == str(secret)):
                    return str(decoded['from'])
                
    def decode_message(self,encoded):
        if('from' in encoded):
            decoded = {}
            decoded['from'] = base64.standard_b64decode(encoded['from'])
            decoded['data'] = base64.standard_b64decode(encoded['data']).decode('ascii')
            decoded['seqno'] = base64.standard_b64decode(encoded['seqno'])
            decoded['topicIDs'] = encoded['topicIDs']
            decoded['encoded'] = encoded
            return decoded
        else:
            return None
        
    def serialize_keras_model(self,model):
        model.save('temp_model.h5')
        f = open('temp_model.h5','rb')
        model_bin = f.read()
        f.close()
        return model_bin
    
    def deserialize_keras_model(self,model_bin):
        g = open('temp_model2.h5','wb')
        g.write(model_bin)
        g.close()
        model = keras.models.load_model('temp_model2.h5')
        return model
    
    def serialize_numpy(self, tensor):
        return json.dumps(tensor.tolist()) # nested lists with same data, indices
    
    def deserialize_numpy(self,json_array):
        return np.array(json.loads(json_array)).astype('float')
    
    def publish(self,channel,dict_message):
        self.api.pubsub_pub(topic=channel,payload=json.dumps(dict_message))
    
    def generate_fit_spec(self, model,input,target,valid_input=None,valid_target=None,batch_size=1,epochs=1,log_interval=1):
        
        model_bin = self.serialize_keras_model(model)
        model_addr = self.api.add_bytes(model_bin)

        train_input = self.serialize_numpy(input)
        train_target = self.serialize_numpy(target)

        if(valid_input is None):
            valid_input = self.serialize_numpy(input)
        else:
            valid_input = self.serialize_numpy(valid_input)

        if(valid_target is None):
            valid_target = self.serialize_numpy(target)
        else:
            valid_target = self.serialize_numpy(valid_target)

        datasets = [train_input,train_target,valid_input,valid_target]
        data_json = json.dumps(datasets)
        data_addr = self.api.add_str(data_json)

        spec = {}
        spec['model_addr'] = model_addr
        spec['data_addr'] = data_addr
        spec['batch_size'] = batch_size
        spec['epochs'] = epochs
        spec['log_interval'] = log_interval
        spec['framework'] = 'keras'
        spec['train_channel'] = 'openmined_train_'+str(model_addr)
        return spec
    
    def listen_to_channel(self,handle_message,channel):
        new_models = self.api.pubsub_sub(topic=channel,stream=True)


        for m in new_models:
            message = self.decode_message(m)
            if(message is not None):
                out = handle_message(message)
                if(out is not None):
                    return out
                
    def keras2ipfs(self,model):
        return self.api.add_bytes(self.serialize_keras_model(model))
    
    def ipfs2keras(self,model_addr):
        model_bin = self.api.cat(model_addr)
        return self.deserialize_keras_model(model_bin)
    
    def receive_model(self,message, verbose=True):
    
        msg = json.loads(message['data'])
        if(msg is not None):
            if(msg['type'] == 'transact'):
                return self.ipfs2keras(msg['model_addr']),msg
            elif(msg['type'] == 'log'):
                if(verbose):
                    output = "Worker:" + msg['worker_id'][-5:]
                    output += " - Epoch " + str(msg['epoch_id']) + " of " + str(msg['num_epochs'])
                    output += " - Valid Loss: " + str(msg['eval_loss'])[0:8]
                    print(output)
    
    def fit_worker(self,message):
    
        decoded = json.loads(message['data'])

        if(decoded['framework'] == 'keras'):

            model = self.ipfs2keras(decoded['model_addr'])

            try:
                np_strings = json.loads(self.api.cat(decoded['data_addr']))
            except:
                print("MUST USE PYTHON VERSION 3.6!!!!")
                assert(False)

            input,target,valid_input,valid_target = list(map(lambda x:self.deserialize_numpy(x),np_strings))
            
            for e in range(0,decoded['epochs'],decoded['log_interval']):
                model.fit(input, target, batch_size=decoded['batch_size'], epochs=decoded['log_interval'], verbose=0)
                eval_loss = model.evaluate(valid_input,valid_target,verbose=0)
                spec = {}
                spec['type'] = 'log'
                spec['eval_loss'] = eval_loss
                spec['epoch_id'] = e
                spec['num_epochs'] = decoded['epochs']
                spec['parent_model'] = decoded['model_addr']
                spec['worker_id'] = self.id
                self.publish(channel=decoded['train_channel'],dict_message=spec)

            spec = {}
            spec['type'] = 'transact'
            spec['model_addr'] = self.keras2ipfs(model)
            spec['eval_loss'] = eval_loss
            spec['parent_model'] = decoded['model_addr']
            spec['worker_id'] = self.id
            self.publish(channel=decoded['train_channel'],dict_message=spec)

            output = "Model:" + spec['parent_model'][-5:]
            output += " - Epochs " + str(decoded['epochs'])
            output += " - Valid Loss: " + str(spec['eval_loss'])[0:8]
            print(output)

            
    def fit(self, model,input,target,valid_input=None,valid_target=None,batch_size=1,epochs=1,log_interval=1,message_handler=None):
    
        if(message_handler is None):
            message_handler = self.receive_model
        spec = self.generate_fit_spec(model,input,target,valid_input,valid_target,batch_size,epochs,log_interval)
        self.publish('openmined',spec)
        
        trained = self.listen_to_channel(message_handler,spec['train_channel'])
        return trained
    
    def work(self):
        self.listen_to_channel(self.fit_worker,'openmined')