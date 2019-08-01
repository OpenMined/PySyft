# syft.frameworks.torch package

## Subpackages

* syft.frameworks.torch.crypto package

  * Submodules

  * syft.frameworks.torch.crypto.securenn module

  * syft.frameworks.torch.crypto.spdz module

  * Module contents

* syft.frameworks.torch.differential_privacy package

  * Submodules

  * syft.frameworks.torch.differential_privacy.pate module

  * Module contents

* syft.frameworks.torch.federated package

  * Submodules

  * syft.frameworks.torch.federated.dataloader module

  * syft.frameworks.torch.federated.dataset module

  * syft.frameworks.torch.federated.utils module

  * Module contents

* syft.frameworks.torch.tensors package

  * Subpackages

    * syft.frameworks.torch.tensors.decorators package

      * Submodules

      * syft.frameworks.torch.tensors.decorators.logging module

      * Module contents

    * syft.frameworks.torch.tensors.interpreters package

      * Submodules

      * syft.frameworks.torch.tensors.interpreters.Polynomial module

      * syft.frameworks.torch.tensors.interpreters.abstract module

      * syft.frameworks.torch.tensors.interpreters.additive_shared module

      * syft.frameworks.torch.tensors.interpreters.multi_pointer module

      * syft.frameworks.torch.tensors.interpreters.native module

      * syft.frameworks.torch.tensors.interpreters.plusisminus module

      * syft.frameworks.torch.tensors.interpreters.pointer module

      * syft.frameworks.torch.tensors.interpreters.precision module

      * Module contents

  * Module contents


## Submodules

## syft.frameworks.torch.functions module


#### syft.frameworks.torch.functions.combine_pointers(\*pointers)
Accepts a list of pointers and returns them as a
MultiPointerTensor. See MultiPointerTensor docs for
details.

Arg:



    ```
    *
    ```

    pointers: a list of pointers to tensors (including

        their wrappers like normal)

## syft.frameworks.torch.hook module


#### class syft.frameworks.torch.hook.TorchHook(torch, local_worker: syft.workers.base.BaseWorker = None, is_client: bool = True, verbose: bool = True)
Bases: `object`

A Hook which Overrides Methods on PyTorch Tensors.

The purpose of this class is to:

    * extend torch methods to allow for the moving of tensors from one

    worker to another.
    \* override torch methods to execute commands on one worker that are
    called on tensors controlled by the local worker.

This class is typically the first thing you will initialize when using
PySyft with PyTorch because it is responsible for augmenting PyTorch with
PySyft’s added functionality (such as remote execution).


* **Parameters**

    * **local_worker** – An optional BaseWorker instance that lets you provide a
      local worker as a parameter which TorchHook will assume to be the
      worker owned by the local machine. If you leave it empty,
      TorchClient will automatically initialize a
      `workers.VirtualWorker` under the assumption you’re looking
      to do local experimentation or development.

    * **is_client** – An optional boolean parameter (default True), indicating
      whether TorchHook is being initialized as an end-user client.This
      can impact whether or not variables are deleted when they fall out
      of scope. If you set this incorrectly on a end user client, Tensors
      and Variables will never be deleted. If you set this incorrectly on
      a remote machine (not a client), tensors will not get saved. It’s
      really only important if you’re not initializing the local worker
      yourself.

    * **verbose** – An optional boolean parameter (default True) to indicate
      whether or not to print the operations as they occur.

    * **queue_size** – An integer optional parameter (default 0) to specify the
      max length of the list that stores the messages to be sent.


### Example

```python
>>> import syft as sy
>>> hook = sy.TorchHook()
Hooking into Torch...
Overloading Complete.
>>> x = sy.Tensor([-2,-1,0,1,2,3])
>>> x
-2
-1
0
1
2
3
[syft.core.frameworks.torch.tensor.FloatTensor of size 6]
```


#### get_hooked_additive_shared_method(attr)
Hook a method to send it multiple recmote workers


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_func(attr)
Hook a function in order to inspect its args and search for pointer
or other syft tensors.
- Calls to this function with normal tensors or numbers / string trigger

> usual behaviour

* Calls with pointers send the command to the location of the pointer(s)

* Calls with syft tensor will in the future trigger specific behaviour


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_method(method_name)
Hook a method in order to replace all args/kwargs syft/torch tensors with
their child attribute if they exist
If so, forward this method with the new args and new self, get response
and “rebuild” the torch tensor wrapper upon all tensors found
If not, just execute the native torch methodn


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_multi_pointer_method(attr)
Hook a method to send it multiple recmote workers


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_pointer_method(attr)
Hook a method to send it to remote worker


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method



#### get_hooked_syft_method(attr)
Hook a method in order to replace all args/kwargs syft/torch tensors with
their child attribute, forward this method with the new args and new self,
get response and “rebuild” the syft tensor wrapper upon all tensors found


* **Parameters**

    **attr** (*str*) – the method to hook



* **Returns**

    the hooked method


## syft.frameworks.torch.hook_args module


#### syft.frameworks.torch.hook_args.build_unwrap_args_with_rules(args, rules, return_tuple=False)
Build a function given some rules to efficiently replace in the args object
syft tensors with their child (but not pointer as they don’t have .child),
and do nothing for other type of object including torch tensors, str,
numbers, bool, etc.
Pointers trigger an error which can be caught to get the location for
forwarding the call.


* **Parameters**

    * **args** (*tuple*) – the arguments given to the function / method

    * **rules** (*tuple*) – the same structure but with boolean, true when there is
      a tensor

    * **return_tuple** (*bool*) – force to return a tuple even with a single element



* **Returns**

    a function that replace syft arg in args with arg.child



#### syft.frameworks.torch.hook_args.build_get_tensor_type(rules, layer=None)
Build a function which uses some rules to find efficiently the first tensor in
the args objects and return the type of its child.


* **Parameters**

    * **rules** (*tuple*) – a skeleton object with the same structure as args but each tensor
      is replaced with a 1 and other types (int, str) with a 0

    * **layer** (*list** or **None*) – keep track of the path of inspection: each element in the list
      stand for one layer of deepness into the object, and its value for the index
      in the current layer. See example for details



* **Returns**

    a function returning a type


### Example

*Understanding the layer parameter*
obj = (a, [b, (c, d)], e)
the layer position is for:
a: [0]
b: [1, 0]
c: [1, 1, 0]
d: [1, 1, 1]
e: [2]

*Global behaviour example*
rules = (0, [1, (0, 0), 0)
- First recursion level

> 0 found -> do nothing
> list found -> recursive call with layer = [1]

* Second recursion level
  1 found -> update layer to [1, 0]

  > build the function x: type(x[1][0])
  > break

* Back to first recursion level
  save the function returned in the lambdas list
  0 found -> do nothing
  exit loop
  return the first (and here unique) function


#### syft.frameworks.torch.hook_args.build_unwrap_args_from_function(args, return_tuple=False)
Build the function f that hook the arguments:
f(args) = new_args


#### syft.frameworks.torch.hook_args.build_wrap_reponse_from_function(response, wrap_type, wrap_args)
Build the function that hook the response.

### Example

p is of type Pointer
f is the hook_response_function
then f(p) = (Wrapper)>Pointer


#### syft.frameworks.torch.hook_args.build_register_response()
Build a function given some rules to efficiently replace in the response object
torch tensors with a pointer after they are registered, and do nothing for other
types of object including , str, numbers, bool, etc.


* **Parameters**

    * **response** – the response

    * **rules** – the rule specifying where the tensors are

    * **return_tuple** – force to return a tuple even with a single element



* **Returns**

    The function to apply on generic responses



#### syft.frameworks.torch.hook_args.build_register_response_function(response: object)
Build the function that registers the response and replaces tensors with pointers.

### Example

(1, tensor([1, 2]) is the response
f is the register_response_function
then f(p) = (1, (Wrapper)>Pointer)


#### syft.frameworks.torch.hook_args.build_response_hook_with_rule(response, rules, wrap_type, wrap_args, return_tuple=False)
Build a function given some rules to efficiently replace in the response object
syft or torch tensors with a wrapper, and do nothing for other types of object
including , str, numbers, bool, etc.


* **Parameters**

    * **response** – a response used to build the hook function

    * **rules** – the same structure objects but with boolean, at true when is replaces
      a tensor

    * **return_tuple** – force to return a tuple even with a single element


Response:

    a function to “wrap” the response


#### syft.frameworks.torch.hook_args.build_rule(args)
Inspect the args object to find torch or syft tensor arguments and
return a rule whose structure is the same as the args object,
with 1 where there was (torch or syft) tensors and 0 when
not (ex: number, str, …)

### Example

in: ([tensor(1, 2), [Pointer@bob](mailto:Pointer@bob)], 42)
out: ([1, 1], 0)


#### syft.frameworks.torch.hook_args.eight_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.five_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.four_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.four_layers(idx1, \*ids)

#### syft.frameworks.torch.hook_args.unwrap_args_from_function(attr, args, kwargs, return_args_type=False)
See unwrap_args_from_method for details


* **Parameters**

    * **attr** (*str*) – the name of the function being called

    * **args** (*list*) – the arguments being passed to the function

    * **kwargs** (*dict*) – the keyword arguments being passed to the function
      (these are not hooked ie replace with their .child attr)

    * **return_args_type** (*bool*) – return the type of the tensors in the

    * **arguments** (*original*) –



* **Returns**

    * the arguments where all tensors are replaced with their child

    * the type of this new child

    (- the type of the tensors in the arguments)




#### syft.frameworks.torch.hook_args.unwrap_args_from_method(attr, method_self, args, kwargs)
Method arguments are sometimes simple types (such as strings or ints) but
sometimes they are custom Syft tensors such as wrappers (torch.Tensor) or LoggingTensor
or some other tensor type. Complex types (which have a .child attribute) need to
have arguments converted from the arg to arg.child so that the types match as the
method is being called down the chain. To make this efficient, we cache which args
need to be replaced with their children in a dictionary called
hook_method_args_functions. However, sometimes a method (an attr) has multiple
different argument signatures, such that sometimes arguments have .child objects
and other times they don’t (such as x.div(), which can accept either a tensor or a
float as an argument). This invalidates the cache, so we need to have a try/except
which refreshes the cache if the signature triggers an error.


* **Parameters**

    * **attr** (*str*) – the name of the method being called

    * **method_self** – the tensor on which the method is being called

    * **args** (*list*) – the arguments being passed to the method

    * **kwargs** (*dict*) – the keyword arguments being passed to the function
      (these are not hooked ie replace with their .child attr)



#### syft.frameworks.torch.hook_args.hook_response(attr, response, wrap_type, wrap_args={}, new_self=None)
When executing a command, arguments are inspected and all tensors are replaced
with their child attribute until a pointer or a torch tensor is found (for
example an argument could be a torch wrapper with a child being a LoggingTensor, with
a child being a torch tensor). When the result of the command is calculated,
we need to rebuild this chain in the reverse order (in our example put back
a LoggingTensor on top of the result and then a torch wrapper).
To make this efficient, we cache which elements of the response (which can be more
complicated with nested tuples for example) need to be wrapped in a dictionary called
hook_method_response_functions. However, sometimes a method (an attr) has multiple
different response signatures. This invalidates the cache, so we need to have a
try/except which refreshes the cache if the signature triggers an error.


* **Parameters**

    * **attr** (*str*) – the name of the method being called

    * **response** (*list** or **dict*) – the arguments being passed to the tensor

    * **wrap_type** (*type*) – the type of wrapper we’d like to have

    * **wrap_args** (*dict*) – options to give to the wrapper (for example the

    * **for the precision tensor****)** (*precision*) –

    * **new_self** – used for the can just below of inplace ops



#### syft.frameworks.torch.hook_args.many_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.one(_args)

#### syft.frameworks.torch.hook_args.one_fold(return_tuple, \*\*kwargs)

#### syft.frameworks.torch.hook_args.one_layer(idx1)

#### syft.frameworks.torch.hook_args.register_response(attr: str, response: object, response_ids: object, owner: syft.workers.abstract.AbstractWorker)
When a remote worker execute a command sent by someone else, the response is
inspected: all tensors are stored by this worker and a Pointer tensor is
made for each of them.

To make this efficient, we cache which elements of the response (which can be more
complicated with nested tuples for example) in the dict register_response_functions

However, sometimes a function  (an attr) has multiple different response signatures.
This invalidates the cache, so we need to have a try/except which refreshes the
cache if the signature triggers an error.


* **Parameters**

    * **attr** (*str*) – the name of the function being called

    * **response** (*object*) – the response of this function

    * **owner** (*BaseWorker*) – the worker which registers the tensors



#### syft.frameworks.torch.hook_args.register_tensor(tensor: Union[torch.Tensor, syft.frameworks.torch.tensors.interpreters.abstract.AbstractTensor], response_ids: List = [], owner: syft.workers.abstract.AbstractWorker = None)
Register a tensor


* **Parameters**

    * **tensor** – the tensor

    * **response_ids** – list of ids where the tensor should be stored
      and each id is pop out when needed

    * **owner** – the owner that makes the registration



* **Returns**

    the pointer



#### syft.frameworks.torch.hook_args.seven_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.six_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.three_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.three_layers(idx1, \*ids)

#### syft.frameworks.torch.hook_args.two_fold(lambdas, args, \*\*kwargs)

#### syft.frameworks.torch.hook_args.two_layers(idx1, idx2)

#### syft.frameworks.torch.hook_args.typed_identity(a)
We need to add typed identity for arguments which can be either number
or tensors. If the argument changes from an int to a tensor, the
assertion error triggered by typed_identity will be caught and a
new signature will be computed for the command.


#### syft.frameworks.torch.hook_args.zero_fold(\*a, \*\*k)
## syft.frameworks.torch.overload_torch module


#### class syft.frameworks.torch.overload_torch.Module()
Bases: `object`


#### class syft.frameworks.torch.overload_torch.Overloaded()
Bases: `object`


#### static overload_function(attr)
hook args and response for functions that hold the @overloaded.function decorator


#### static overload_method(attr)
hook args and response for methods that hold the @overloaded.method decorator


#### static overload_module(attr)
## syft.frameworks.torch.torch_attributes module


#### class syft.frameworks.torch.torch_attributes.TorchAttributes(torch: module, hook: module)
Bases: `object`

Adds torch module related custom attributes.

TorchAttributes is a special class where all custom attributes related
to the torch module can be added. Any global parameter, configuration,
or reference relating to PyTorch should be stored here instead of
attaching it directly to some other part of the global namespace.

The main reason we need this is because the hooking process occasionally
needs to save global objects, notably including what methods to hook and
what methods to NOT hook.

This will hold all necessary attributes PySyft needs.


* **Parameters**

    * **torch** – A ModuleType indicating the torch module

    * **hook** – A ModuleType indicating the modules to hook



#### static apply_fix16922(torch)
Apply the fix made in PR16922 of PyTorch until people use PyTorch 1.0.2
:param torch: the pytorch module


#### eval_torch_modules()
Builds a mapping between the hooked and native commands.

For each torch command functions in native_commands, transform the
dictionary so that to each key, which is the name of the hooked
command, now corresponds a value which is the evaluated native name of
the command, namely the native command.

Note that we don’t do this for methods.


#### static get_native_torch_name(attr: str)
Returns the name of the native command for the given hooked command.


* **Parameters**

    **attr** – A string indicating the hooked command name (ex: torch.add)



* **Returns**

    torch.native_add)



* **Return type**

    The name of the native command (ex



#### is_inplace_method(method_name)
Says if a method is inplace or not by test if it ends by _ and is not a __xx__
:param method_name: the name for the method
:return: boolean

## Module contents
