# syft.frameworks.torch.tensors.interpreters package

## Submodules

## syft.frameworks.torch.tensors.interpreters.Polynomial module


#### class syft.frameworks.torch.tensors.interpreters.Polynomial.PolynomialTensor(function=<built-in method tensor of type object>, precision=10)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`

Tensor type to provide non-linear function approximations

MPC and Homomorphic Encryption are capable of performing some addition and logical operations.
Non-Linear functions could be approximated as a series of approximated functions of basic arithmetic
operations using function approximations such as interpolation/Taylor series.

The polynomial tensor provides flexibility to consider every non-linear function as piecewise linear function
and fit over different intervals.


* **Parameters**

    * **function****[****callable****,****Optional****]** – Function to applied to function approximation coefficients. Used to encrypt coefficients.

    * **precision****[****integer****]** – Precision of approximated values



#### addfunction(name, degree, piecewise, function)
Add function to function_attr dictionary.


* **Parameters**

    * **name****[****str****]** – Name of function

    * **degree****[****int****]** – Degree of function

    * **piecewise****[****List****]** – List of piecewise functions in format [min_val of fit,max_val of fit,step of fit,function to fit values]

    * **function****[****callable****]** – Base function



#### applycoefs(polyinstance, function)
Apply a given function over Numpy interpolation instances.This function could be used to encrypt coefficients of function approximations approximated using interpolation

Parameters:

polyinstance (Numpy poly1d) : Interpolation instance
function (Callable) : Function to be applied


#### defaultfunctions()
Initializes default function approximations exp,log,sigmoid and tanh


#### exp(x)
Method provides exponential function approximation interms of Taylor Series

Parameters:

x: Torch tensor

return:

approximation of the sigmoid function as a torch tensor


#### fit_function(name, min_val=0, max_val=10, steps=100, degree=10)
Interpolated approximation of given function

name: Name of function as defined in self.setting
min_val: Minimum range of interpolation fit
max_val: Maximum range of interpolation fit
steps:   Steps of interpolation fit
degree: Degree of interpolation fit
function: The function used to encrypt function approximation coefficients


* **Returns**

    Approximated function



* **Return type**

    f_interpolated (Numpy Poly1d)



#### get_val(name, x)
Get value of given function approximation


* **Parameters**

    * **name****[****str****]** – Name of function

    * **tensor****,****float****,****integer****]** (*value**[**torch*) – Value to be approximated



* **Returns**

    Approximated value using given function approximation



#### interpolate(function: Callable, interval: List[Union[int, float]], degree: int = 10)
Returns a interpolated version of given function using Numpy’s polyfit method

> Args:

> > function (a lambda function): Base function to be approximated
> > interval (list of floats/integers): Interval of values to be approximated
> > degree (Integer): Degree of polynomial approximation
> > precision(Integer): Precision of coefficients


* **Returns**

    Approximated Function



* **Return type**

    f_interpolated (Numpy poly1d)



#### piecewise_linear_eval(data, x)
Get approximated value for a given function. This takes only scalar value.
If you have a Numpy array or torch tensor consider passing it using a
lambda or

```
torch.apply_
```

 method.


* **Parameters**

    * **List****]** (*data**[**2D*) – Instance of piecewise linear fit taking values [min_val,max_val,function approximation method]

    * **or Integer****]** (*x**[**Float*) – Value to be approximated



#### piecewise_linear_fit(name, array)
Fit a piecewise linear function. This can be used to approximate a non-linear function as seperate linear functions valid for seperate ranges.
For , instance function approximations are more accurate for exponential when seperate instances  of interpolation are fit between
-10 to 0 and 0 to 10.


* **Parameters**

    **List****]** (*array**[**2D*) – Each instance of list must take four values [min_val,steps,max_val,function approximation method]



* **Returns**

    Each instance of list with four values [min_val,max_val,Approximated function]



* **Return type**

    array[2D List]



#### sigmoid(x)
Parameters:

Method provides Sigmoid function approximation interms of Taylor Series

x: Torch tensor

return:

approximation of the sigmoid function as a torch tensor

## syft.frameworks.torch.tensors.interpreters.abstract module


#### class syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor(id: int = None, owner: Optional[syft.workers.abstract.AbstractWorker] = None, tags: List[str] = None, description: str = None, child=None)
Bases: `abc.ABC`

This is the tensor abstraction.


#### get_class_attributes()
Return all elements which defines an instance of a certain class.
By default there is nothing so we return an empty dict, but for
example for fixed precision tensor, the fractional precision is
very important.


#### classmethod handle_func_command(command)
Receive an instruction for a function to be applied on a Syft Tensor,
Replace in the args all the LogTensors with
their child attribute, forward the command instruction to the
handle_function_command of the type of the child attributes, get the
response and replace a Syft Tensor on top of all tensors found in
the response.


* **Parameters**

    * **command** – instruction of a function command: (command name,

    * **self>****, ****arguments****[****, ****kwargs****]****)** (*<no*) –



* **Returns**

    the response of the function command



#### is_wrapper( = False)

#### on(tensor: syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor, wrap: bool = True)
Add a syft(log) tensor on top of the tensor.


* **Parameters**

    * **tensor** – the tensor to extend

    * **wrap** – if true, add the syft tensor between the wrapper

    * **the rest of the chain. If false****, ****just add it at the top** (*and*) –



* **Returns**

    a syft/torch tensor



#### classmethod on_function_call(\*args)
Override this to perform a specific action for each call of a torch
function with arguments containing syft tensors of the class doing
the overloading


#### classmethod rgetattr(obj, attr, \*args)
Get an attribute recursively


* **Parameters**

    * **obj** – the object holding the attribute

    * **attr** – nested attribute

    * **args** – optional arguments to provide



* **Returns**

    the attribute obj.attr


### Example

```python
>>> rgetattr(obj, 'attr1.attr2.attr3')
[Out] obj.attr1.attr2.attr3
```


#### ser(\*args, \*\*kwargs)

#### serialize()
Serializes the tensor on which it’s called.

This is the high level convenience function for serializing torch
tensors. It includes three steps, Simplify, Serialize, and Compress as
described in serde.py.
By default serde is compressing using LZ4


* **Returns**

    The serialized form of the tensor.
    For example:

    > x = torch.Tensor([1,2,3,4,5])
    > x.serialize() # returns a serialized object




#### shape()

#### wrap()
Wraps the class inside torch tensor.

Because PyTorch does not (yet) support functionality for creating
arbitrary Tensor types (via subclassing torch.Tensor), in order for our
new tensor types (such as PointerTensor) to be usable by the rest of
PyTorch (such as PyTorch’s layers and loss functions), we need to wrap
all of our new tensor types inside of a native PyTorch type.

This function adds a .wrap() function to all of our tensor types (by
adding it to AbstractTensor), such that (on any custom tensor
my_tensor), my_tensor.wrap() will return a tensor that is compatible
with the rest of the PyTorch API.


* **Returns**

    A pytorch tensor.



#### syft.frameworks.torch.tensors.interpreters.abstract.initialize_tensor(hook_self, cls, torch_tensor: bool = False, owner=None, id=None, \*init_args, \*\*init_kwargs)
Initializes the tensor.


* **Parameters**

    * **hook_self** – A reference to TorchHook class.

    * **cls** – An object to keep track of id, owner and whether it is a native
      tensor or a wrapper over pytorch.

    * **torch_tensor** – A boolean parameter (default False) to indicate whether
      it is torch tensor or not.

    * **owner** – The owner of the tensor being initialised, leave it blank
      to if you have already provided a reference to TorchHook class.

    * **id** – The id of tensor, a random id will be generated if there is no id
      specified.


## syft.frameworks.torch.tensors.interpreters.additive_shared module


#### class syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor(shares: dict = None, owner=None, id=None, field=None, n_bits=None, crypto_provider=None, tags=None, description=None)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`


#### add(\*args, \*\*kwargs)

#### argmax(dim=None)

#### static dispatch(args, worker)
utility function for handle_func_command which help to select
shares (seen as elements of dict) in an argument set. It could
perhaps be put elsewhere


* **Parameters**

    * **args** – arguments to give to a functions

    * **worker** – owner of the shares to select



* **Returns**

    args where the AdditiveSharedTensors are replaced by
    the appropriate share



#### eq(other)

#### ge(other)

#### static generate_shares(secret, n_workers, field, random_type)
The cryptographic method for generating shares given a secret tensor.


* **Parameters**

    * **secret** – the tensor to be shared.

    * **n_workers** – the number of shares to generate for each value
      (i.e., the number of tensors to return)

    * **field** – 1 + the max value for a share

    * **random_type** – the torch type shares should be encoded in (use the smallest possible
      given the choise of mod”



#### get()
Fetches all shares and returns the plaintext tensor they represent


#### get_class_attributes()
Specify all the attributes need to build a wrapper correctly when returning a response,
for example precision_fractional is important when wrapping the result of a method
on a self which is a fixed precision tensor with a non default precision_fractional.


#### gt(other)

#### classmethod handle_func_command(command)
Receive an instruction for a function to be applied on a Syft Tensor,
Replace in the args all the LogTensors with
their child attribute, forward the command instruction to the
handle_function_command of the type of the child attributes, get the
response and replace a Syft Tensor on top of all tensors found in
the response.


* **Parameters**

    * **command** – instruction of a function command: (command name,

    * **self>****, ****arguments****[****, ****kwargs****]****)** (*<no*) –



* **Returns**

    the response of the function command



#### init_shares(\*owners)
Initializes shares and distributes them amongst their respective owners


* **Parameters**

    **the list of shareholders. Can be of any length.** (*\*owners*) –



#### le(other)

#### locations()
Provide a locations attribute


#### lt(other)

#### matmul(other)
Multiplies two tensors matrices together


* **Parameters**

    * **self** – an AdditiveSharingTensor

    * **other** – another AdditiveSharingTensor or a MultiPointerTensor



#### max(dim=None, return_idx=False)
Return the maximum value of an additive shared tensor


* **Parameters**

    * **dim** (*None** or **int*) – if not None, the dimension on which
      the comparison should be done

    * **return_idx** (*bool*) – Return the index of the maximum value
      Note the if dim is specified then the index is returned
      anyway to match the Pytorch syntax.



* **Returns**

    the maximum value (possibly across an axis)
    and optionally the index of the maximum value (possibly across an axis)



#### mm(\*args, \*\*kwargs)
Multiplies two tensors matrices together


#### mod(\*args, \*\*kwargs)

#### mul(other)
Multiplies two tensors together


* **Parameters**

    * **self** (*AdditiveSharingTensor*) – an AdditiveSharingTensor

    * **other** – another AdditiveSharingTensor, or a MultiPointerTensor, or an integer



#### positive()

#### reconstruct()
Reconstruct the shares of the AdditiveSharingTensor remotely without
its owner being able to see any sensitive value


* **Returns**

    A MultiPointerTensor where all workers hold the reconstructed value



#### relu()

#### shape()
Return the shape which is the shape of any of the shares


#### sub(\*args, \*\*kwargs)

#### torch( = <syft.frameworks.torch.overload_torch.Module object>)

#### virtual_get()
Get the value of the tensor without calling get
- Useful for debugging, only for VirtualWorkers

## syft.frameworks.torch.tensors.interpreters.multi_pointer module


#### class syft.frameworks.torch.tensors.interpreters.multi_pointer.MultiPointerTensor(location: syft.workers.base.BaseWorker = None, id_at_location: Union[str, int] = None, register: bool = False, owner: syft.workers.base.BaseWorker = None, id: Union[str, int] = None, garbage_collect_data: bool = True, point_to_attr: str = None, tags: List[str] = None, description: str = None, children: List[syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor] = [])
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`


#### static dispatch(args, worker)
utility function for handle_func_command which help to select
shares (seen as elements of dict) in an argument set. It could
perhaps be put elsewhere


* **Parameters**

    * **args** – arguments to give to a functions

    * **worker** – owner of the shares to select



* **Returns**

    args where the MultiPointerTensor are replaced by
    the appropriate share



#### get(sum_results: bool = False)

#### classmethod handle_func_command(command)
Receive an instruction for a function to be applied on a Syft Tensor,
Replace in the args all the LogTensors with
their child attribute, forward the command instruction to the
handle_function_command of the type of the child attributes, get the
response and replace a Syft Tensor on top of all tensors found in
the response.


* **Parameters**

    * **command** – instruction of a function command: (command name,

    * **self>****, ****arguments****[****, ****kwargs****]****)** (*<no*) –



* **Returns**

    the response of the function command



#### shape()
This method returns the shape of the data being pointed to.
This shape information SHOULD be cached on self._shape, but
occasionally this information may not be present. If this is the
case, then it requests the shape information from the remote object
directly (which is inefficient and should be avoided).


#### virtual_get(sum_results: bool = False)
Get the value of the tensor without calling get - Only for VirtualWorkers

## syft.frameworks.torch.tensors.interpreters.native module


#### class syft.frameworks.torch.tensors.interpreters.native.TorchTensor(id: int = None, owner: Optional[syft.workers.abstract.AbstractWorker] = None, tags: List[str] = None, description: str = None, child=None)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`

Add methods to this tensor to have them added to every torch.Tensor object.

This tensor is simply a more convenient way to add custom functions to
all Torch tensor types. When you add a function to this tensor, it will
be added to EVERY native torch tensor type (i.e. torch.Torch) automatically
by the TorchHook (which is in frameworks/torch/hook.py).

Note: all methods from AbstractTensor will also be included because this
tensor extends AbstractTensor. So, if you’re looking for a method on
the native torch tensor API but it’s not listed here, you might try
checking AbstractTensor.


#### attr(attr_name)

#### combine(\*pointers)
This method will combine the child pointer with another list of pointers


* **Parameters**

    **a list of pointers to be combined into a MultiPointerTensor** (*\*pointers*) –



#### copy()

#### create_pointer(location: syft.workers.base.BaseWorker = None, id_at_location: str = None, register: bool = False, owner: syft.workers.base.BaseWorker = None, ptr_id: str = None, garbage_collect_data: bool = True, shape=None)
Creates a pointer to the “self” torch.Tensor object.

This method is called on a torch.Tensor object, returning a pointer
to that object. This method is the CORRECT way to create a pointer,
and the parameters of this method give all possible attributes that
a pointer can be created with.


* **Parameters**

    * **location** – The BaseWorker object which points to the worker on which
      this pointer’s object can be found. In nearly all cases, this
      is self.owner and so this attribute can usually be left blank.
      Very rarely you may know that you are about to move the Tensor
      to another worker so you can pre-initialize the location
      attribute of the pointer to some other worker, but this is a
      rare exception.

    * **id_at_location** – A string or integer id of the tensor being pointed
      to. Similar to location, this parameter is almost always
      self.id and so you can leave this parameter to None. The only
      exception is if you happen to know that the ID is going to be
      something different than self.id, but again this is very rare
      and most of the time, setting this means that you are probably
      doing something you shouldn’t.

    * **register** – A boolean parameter (default False) that determines
      whether to register the new pointer that gets created. This is
      set to false by default because most of the time a pointer is
      initialized in this way so that it can be sent to someone else
      (i.e., “Oh you need to point to my tensor? let me create a
      pointer and send it to you” ). Thus, when a pointer gets
      created, we want to skip being registered on the local worker
      because the pointer is about to be sent elsewhere. However, if
      you are initializing a pointer you intend to keep, then it is
      probably a good idea to register it, especially if there is any
      chance that someone else will initialize a pointer to your
      pointer.

    * **owner** – A BaseWorker parameter to specify the worker on which the
      pointer is located. It is also where the pointer is registered
      if register is set to True.

    * **ptr_id** – A string or integer parameter to specify the id of the pointer
      in case you wish to set it manually for any special reason.
      Otherwise, it will be set randomly.

    * **garbage_collect_data** – If true (default), delete the remote tensor when the
      pointer is deleted.



* **Returns**

    A torch.Tensor[PointerTensor] pointer to self. Note that this
    object will likely be wrapped by a torch.Tensor wrapper.



#### data()

#### describe(description)

#### description()

#### enc_fix_prec()

#### fix_prec(\*args, \*\*kwargs)

#### fix_prec_(\*args, \*\*kwargs)

#### fix_precision(\*args, \*\*kwargs)

#### fix_precision_(\*args, \*\*kwargs)

#### float_prec()

#### float_prec_()

#### float_precision()

#### float_precision_()

#### get(\*args, inplace: bool = False, \*\*kwargs)
Requests the tensor/chain being pointed to, be serialized and return
:param args: args to forward to worker
:param inplace: if true, return the same object instance, else a new wrapper
:param kwargs: kwargs to forward to worker


#### get_(\*args, \*\*kwargs)
Calls get() with inplace option set to True


#### classmethod handle_func_command(command)
Operates as a router for functions. A function call always starts
by being handled here and 3 scenarii must be considered:

Real Torch tensor:

    The arguments of the function are real tensors so we should
    run the native torch command

Torch wrapper:

    The arguments are just wrappers at the top of a chain
    (ex: wrapper>LoggingTensor>Torch tensor), so just forward
    the instruction to the next layer type in the chain (in
    the example above to LoggingTensor.handle_func_command),
    get the response and replace a wrapper on top of all tensors
    found in the response.

Syft Tensor:

    The arguments are syft tensors of same type: this can happen
    if at any node of the chain where some function is forwarded,
    the handle_func_command modify the function and make a new
    call but keeps the arguments “un-wrapped”. Making a new call
    means that by default the command is treated here in the
    global router.


* **Parameters**

    **command** – instruction of a function command: (command name,


<no self>, arguments[, kwargs])
:return: the response of the function command


#### has_child()

#### id()

#### mid_get()
This method calls .get() on a child pointer and correctly registers the results


#### move(location)

#### remote_get()
Assuming .child is a PointerTensor, this method calls .get() on the tensor
that the .child is pointing to (which should also be a PointerTensor)

TODO: make this kind of message forwarding generic?


#### send(\*location, inplace: bool = False)
Gets the pointer to a new remote object.

One of the most commonly used methods in PySyft, this method serializes
the object upon which it is called (self), sends the object to a remote
worker, creates a pointer to that worker, and then returns that pointer
from this function.


* **Parameters**

    * **location** – The BaseWorker object which you want to send this object
      to. Note that this is never actually the BaseWorker but instead
      a class which instantiates the BaseWorker abstraction.

    * **inplace** – if true, return the same object instance, else a new wrapper



* **Returns**

    A torch.Tensor[PointerTensor] pointer to self. Note that this
    object will likely be wrapped by a torch.Tensor wrapper.



#### send_(\*location)
Calls send() with inplace option, but only with a single location
:param location: workers locations
:return:


#### shape()

#### share(\*owners, field=None, crypto_provider=None)
This is a pass through method which calls .share on the child.


* **Parameters**

    * **owners** (*list*) – a list of BaseWorker objects determining who to send shares to

    * **field** (*int** or **None*) – the arithmetic field where live the shares

    * **crypto_provider** (*BaseWorker** or **None*) – the worker providing the crypto primitives



#### share_(\*args, \*\*kwargs)
Allows to call .share() as an inplace operation


#### tag(\*_tags)

#### tags()
## syft.frameworks.torch.tensors.interpreters.plusisminus module

## syft.frameworks.torch.tensors.interpreters.pointer module


#### class syft.frameworks.torch.tensors.interpreters.pointer.PointerTensor(location: BaseWorker = None, id_at_location: Union[str, int] = None, owner: BaseWorker = None, id: Union[str, int] = None, garbage_collect_data: bool = True, shape: torch.Size = None, point_to_attr: str = None, tags: List[str] = None, description: str = None)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`

A pointer to another tensor.

A PointerTensor forwards all API calls to the remote.PointerTensor objects
point to tensors (as their name implies). They exist to mimic the entire
API of a normal tensor, but instead of computing a tensor function locally
(such as addition, subtraction, etc.) they forward the computation to a
remote machine as specified by self.location. Specifically, every
PointerTensor has a tensor located somewhere that it points to (they should
never exist by themselves). Note that PointerTensor objects can point to
both torch.Tensor objects AND to other PointerTensor objects. Furthermore,
the objects being pointed to can be on the same machine or (more commonly)
on a different one. Note further that a PointerTensor does not know the
nature how it sends messages to the tensor it points to (whether over
socket, http, or some other protocol) as that functionality is abstracted
in the BaseWorker object in self.location.

### Example

```python
>>> import syft as sy
>>> hook = sy.TorchHook()
>>> bob = sy.VirtualWorker(id="bob")
>>> x = sy.Tensor([1,2,3,4,5])
>>> y = sy.Tensor([1,1,1,1,1])
>>> x_ptr = x.send(bob) # returns a PointerTensor, sends tensor to Bob
>>> y_ptr = y.send(bob) # returns a PointerTensor, sends tensor to Bob
>>> # executes command on Bob's machine
>>> z_ptr = x_ptr + y_ptr
```


#### attr(attr_name)

#### data()

#### classmethod find_a_pointer(command)
Find and return the first pointer in the args object, using a trick
with the raising error RemoteTensorFoundError


#### get(deregister_ptr: bool = True)
Requests the tensor/chain being pointed to, be serialized and return

Since PointerTensor objects always point to a remote tensor (or chain
of tensors, where a chain is simply a linked-list of tensors linked via
their .child attributes), this method will request that the tensor/chain
being pointed to be serialized and returned from this function.

**NOTE**: This will typically mean that the remote object will be
removed/destroyed. To just bring a copy back to the local worker,
call .copy() before calling .get().


* **Parameters**

    **deregister_ptr** (*bool**, **optional*) – this determines whether to
    deregister this pointer from the pointer’s owner during this
    method. This defaults to True because the main reason people use
    this method is to move the tensor from the remote machine to the
    local one, at which time the pointer has no use.



* **Returns**

    An AbstractTensor object which is the tensor (or chain) that this
    object used to point to on a remote machine.


TODO: add param get_copy which doesn’t destroy remote if true.


#### get_shape()
Request information about the shape to the remote worker


#### grad()

#### classmethod handle_func_command(command)
Receive an instruction for a function to be applied on a Pointer,
Get the remote location to send the command, send it and get a
pointer to the response, return.
:param command: instruction of a function command: (command name,
None, arguments[, kwargs])
:return: the response of the function command


#### is_none()

#### shape()
This method returns the shape of the data being pointed to.
This shape information SHOULD be cached on self._shape, but
occasionally this information may not be present. If this is the
case, then it requests the shape information from the remote object
directly (which is inefficient and should be avoided).


#### share(\*args, \*\*kwargs)
Send a command to remote worker to additively share a tensor


* **Returns**

    A pointer to an AdditiveSharingTensor


## syft.frameworks.torch.tensors.interpreters.precision module


#### class syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor(owner=None, id=None, field: int = 4611686018427387903, base: int = 10, precision_fractional: int = 3, kappa: int = 1, tags: set = None, description: str = None)
Bases: `syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor`


#### add(\*args, \*\*kwargs)

#### eq(\*args, \*\*kwargs)

#### fix_precision()
This method encodes the .child object using fixed precision


#### float_precision()
this method returns a new tensor which has the same values as this
one, encoded with floating point precision


#### get()
Just a pass through. This is most commonly used when calling .get() on a
FixedPrecisionTensor which has also been shared.


#### get_class_attributes()
Specify all the attributes need to build a wrapper correctly when returning a response,
for example precision_fractional is important when wrapping the result of a method
on a self which is a fixed precision tensor with a non default precision_fractional.


#### classmethod handle_func_command(command)
Receive an instruction for a function to be applied on a FixedPrecision Tensor,
Perform some specific action (like logging) which depends of the
instruction content, replace in the args all the FPTensors with
their child attribute, forward the command instruction to the
handle_function_command of the type of the child attributes, get the
response and replace a FixedPrecision on top of all tensors found in
the response.
:param command: instruction of a function command: (command name,
<no self>, arguments[, kwargs])
:return: the response of the function command


#### matmul(\*args, \*\*kwargs)
Hook manually matmul to add the truncation part which is inherent to multiplication
in the fixed precision setting


#### mul(\*args, \*\*kwargs)
Hook manually mul to add the truncation part which is inherent to multiplication
in the fixed precision setting


#### share(\*owners, field=None, crypto_provider=None)

#### t(\*args, \*\*kwargs)

#### torch( = <syft.frameworks.torch.overload_torch.Module object>)

#### truncate(precision_fractional)
## Module contents
