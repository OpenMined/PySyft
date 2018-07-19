import torch
import json

class TorchGuard():

    def __init__(self):

        # Safety checks for serializing and deserializing torch objects
        # Desperately needs stress testing before going out in the wild
        self.map_tensor_type = {
            'torch.FloatTensor': torch.FloatTensor,
            'torch.DoubleTensor': torch.DoubleTensor,
            'torch.HalfTensor': torch.HalfTensor,
            'torch.ByteTensor': torch.ByteTensor,
            'torch.CharTensor': torch.CharTensor,
            'torch.ShortTensor': torch.ShortTensor,
            'torch.IntTensor': torch.IntTensor,
            'torch.LongTensor': torch.LongTensor
        }
        self.map_var_type = {
            'torch.autograd.variable.Variable': torch.autograd.variable.Variable,
            'torch.nn.parameter.Parameter': torch.nn.parameter.Parameter
        }
        self.map_torch_type = dict(self.map_tensor_type, **self.map_var_type)

    def types_guard(self, torch_type_str):
        """types_guard(torch_type_str) -> torch.Tensor or torch.autograd.Variable

        This method converts strings into a type reference. This prevents
        deserialized JSON from being able to instantiate objects of arbitrary
        type which would be a security concern.

        :Parameters:

        * **torch_type_str (string)** A string representing the type of object that is
          to be returned.

        * **out (a torch type)** The type the string refersto (if it's present in the
          acceptible list self.map_torch_type)
        """
        try:
            return self.map_torch_type[torch_type_str]
        except KeyError:
            raise TypeError(
                "Tried to receive a non-Torch object of type {}.".format(
                    torch_type_str))


    def tensor_contents_guard(self,contents):
        """tensor_contents_guard(contents) -> contents
        check to make sure the incoming list isn't dangerous to use for
               constructing a tensor (likely non-trivial). Accepts the list of JSON objects
               and returns the list of JSON ojects.Throws an exception if there's a
               security concern.
        """
        
        is_instance = lambda tensor : all(isinstance(digit, (int, float)) for digit in tensor)
        assert type(contents) in (list,tuple) , "A list of JSON objects is expected"
        for json_object in contents : 
            try :
                json_object = json.loads(json_object)
            except Exception as e:
                print(e)
            assert (len(json_object) == 1) , "Expects JSON as { 'Tensor' : ( [Values] ,[Values] , ... ) }"
            Tensor_Values = list(json_object.values())
            for tensor in Tensor_Values[0]:
                if is_instance(tensor) == False:
                    raise Exception("Values are expected to be in dtype int and float")
                    break
            tensor_length = map(len,Tensor_Values[0])
            assert len(set(tensor_length)) == 1 , "Tensor dimensions are not same, can't be stacked" 
        return contents

