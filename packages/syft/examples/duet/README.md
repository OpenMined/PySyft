<h1 align="center">
  <br>
  <a href="http://duet.openmined.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/logo_duet.png" alt="PySyft" width="350"></a>
  <br>
  <br>
  Syft in a Notebook
</h1>
<div align="center">
  <a href=""><img src="https://github.com/OpenMined/PySyft/workflows/Tests/badge.svg?branch=main" /></a> <a href=""><img src="https://github.com/OpenMined/PySyft/workflows/Tutorials/badge.svg" /></a> <a href="https://openmined.slack.com/messages/lib_pysyft"><img src="https://img.shields.io/badge/chat-on%20slack-7A5979.svg" /></a> <a href="https://mybinder.org/v2/gh/OpenMined/PySyft/main"><img src="https://mybinder.org/badge.svg" /></a> <a href="http://colab.research.google.com/github/OpenMined/PySyft/blob/main"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a><br /><br />
</div>

<h2 align="center">
  <a href="http://duet.openmined.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/mini_notebooks.png" alt="Duet" width="800"></a>
  <br>
   🎸 Data Owner ⚡️ Data Scientist 🥁
  <br>
</h2>

# Duet 🎸🥁

Duet is the latest part of the Syft family and is designed to provide you with a seamless experience, creating machine learning models in tools you are already familiar with, like Jupyter notebooks and the PyTorch API; while allowing training over a remote session, on data you cannot see, anywhere in the world 🌏.

## Version Support

We support **Linux**, **MacOS** and **Windows** and the following Python and Torch versions.
Older versions may work, however we have stopped testing and supporting them.

| Py / Torch | 1.6 | 1.7 | 1.8 |
| ---------- | --- | --- | --- |
| 3.7        | ✅  | ✅  | ✅  |
| 3.8        | ✅  | ✅  | ✅  |
| 3.9        | ➖  | ✅  | ✅  |

## Setup 🐍

```bash
$ pip install syft
```

## Connecting ⚡️

To connect two Duet sessions, choose one user as the role of Data Owner and the other as the role of Data Scientist. While not necessary, it is usually a good idea to jump on a Video Call with
screen sharing enabled so you can collaborate better.

### Step 1 - Create a Duet

The Data Owner can create a duet in 2 lines of code. After running `sy.duet()` or `sy.launch_duet()`
instructions will be provided as well as a handy code snippet to send to your Data Science partner.

```python
import syft as sy
duet = sy.duet()
```

### Step 2 - Join a Duet

Take the code the Data Owner sends you and run it in your own notebook.

```python
import syft as sy
duet = sy.duet("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

It will give you back a code to send to your Data Owner partner so they can link the sessions.

### Host a Network

The connections are Peer-to-peer but to work around NAT / Firewalls we use WebRTC and STUN.
This requires an initial connection to our OpenGrid server, however, after the connection, traffic is only sent directly between you and your Duet partner. If you have issues with the builtin
session matcher, or simply want to run your own, you can do so with the `syft-network` command:

```
$ syft-network
```

Then simply update your `sy.duet()` commands with the keyword arg: `network_url=http://localhost:5000`
You can also start `syft-network` in IPV6 mode `syft-network IPV6` or by exporting the environment variable `IP_MODE=IPV6`

## Quick Start 📖

### torch.Tensor

There are three main methods which are automatically added to torch.Tensor.

- `torch.Tensor.tag()`
  This lets you apply a string to make it easy to identify a tensor in the store or for your
  partner to search for and get a reference to it.
- `torch.Tensor.describe()`
  This provides for more room to add a lengthy description of your data.
- `torch.Tensor.send()`
  This method lets you send the tensor to Duet. If you want to make it pointable by
  your partner don't forget to use the `pointable=True` keyword argument.

### duet.store

The duet store is where both the Data Owner and Data Scientist can view what is in the
store and get references to those objects as pointers.

- `duet.store.pandas`
  This lets you easily view a pandas table of the contents of the store.
- `duet.store[key]`
  This lets you use the `UID` or unique `tag` to get a pointer to that object.

### TensorPointer

When you get a Pointer to a remote object it will be called `f"{__class__}Pointer"`.
The main one you will interact with is TensorPointer. If you look at the `dir()` of a
Pointer you can see the methods and attributes available to use with it. The return
type of the method or property will return another pointer to that return value. As
many methods on `torch.Tensor` return a `torch.Tensor` you can simply chain methods
which will execute on the next returning TensorPointer.

If you finally want to be able to get the real value behind a pointer you need to
issue a `pointer.get()`.

- `pointer.get()` Will return the value if you have permission or throw an
  AuthorizationException if you do not. If you do not have permission you can request it
  with a `pointer.request()` however, this is non-blocking. For situations where you want
  to build control structure around approval in your code, such as a training loop,
  you should use the `pointer.get(request_block=True)` API. There are many additional
  parameters that you can use to customize your request with `names`, `reasons`,
  `timeout_secs`. Note `.get()` normally deletes the remote object after it has been
  retrieved, you can either use the `delete_obj` param to toggle this or simply use the
  `pointer.get_copy()` convenience method.

### duet.requests

On the Data Owners side, you can use the `duet.requests` API to view, accept, reject and
create auto-responder handlers for requests.

- `duet.requests.pandas`
  This will show a pandas table of the current pending requests.
- `duet.requests[index]`
  This will get the request at the specific index after which you can choose to `req.accept()`
  or `req.deny()`.
- `duet.requests.add_handler()`
  This method provides an API for setting up request handlers which will auto-respond to
  your Duet partners requests. You can use keyword parameters like `name`, `action`,
  `timeout_secs`, `element_quota`, `print_local`, `log_local` to do interesting things like
  setting up a handler to reject all requests for training loss while printing the results
  in your notebook. Or providing a limited `element_quota` for a particular request name
  so that you can be sure only a small amount of data can be viewed by your partner.
- `duet.requests.handlers`
  This property will show you your current active handlers.
- `duet.requests.clear_handlers()`
  This method allows you to quickly and easily remove all your current active handlers.

### sy.Module

To allow the construction of custom models we have created a helper class called sy.Module
which acts very similar to nn.Module.

```python
class SyNet(sy.Module):
    def __init__(self, torch_ref):
        super(SyNet, self).__init__(torch_ref=torch_ref)
        self.ll1 = self.torch_ref.nn.Linear(D_in, H)
        self.ll2 = self.torch_ref.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.ll1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = self.ll2(x)
        return self.torch_ref.nn.functional.log_softmax(x, dim=1)

model = SyNet(torch_ref=torch)
```

The key difference is that you need to pass in a reference to `torch`. Internally the
sy.Module will keep track of the layers and allow you to call `.send()` and `.get()`
on your model at any time. To allow this to work make sure that inside your class
definition you always use the passed in `torch_ref` so that sy.Module can switch this
depending on the context in which the model is running. If you want to know where your
current model variable is used `model.is_local` to check.

### Save and Load

You can save and load your sy.Modules just the same way you would with a torch model
with `model.save()` and `model.load()` or `model.load_state_dict()`. You can even
load weights from a normal PyTorch model you have trained separately as long as the
layers and architecture are the same.

### .get Permissions

One thing to note is that as a Data Scientist you might send data or models to your
Data Owner partner and initially have the permission to `.get()` them back.
However the moment your data or models come into contact with private data on the
Data Owners side they will have their permissions limited until you make a
`.request() or .get(request_block=True)`. This is how model downloading works and
you should expect to require a permission request before being able to download a
model trained on data hosted by your Data Owner partner.

### PyTorch API

We currently support most of the operations and layers within PyTorch 1.5 - 1.7.
However, there are some ops which are either insecure or not appropriate for Duet
or are still in a state of development.

## Example Notebooks 📚

The following examples were taken from the official [PyTorch examples](https://github.com/pytorch/examples) repo.

Each example is split into two notebooks, one for the data owner who wishes to protect their valuable and private training data and one for the data scientist who has some problem they wish to solve and a test set they can use to determine if the data owner's data will help. Together they are able to collaborate over two notebooks to construct and train models, evaluate and share metrics upon request and do inference on a test or individual data items.

| Example                                                          | Data Owner                                                                             | Data Scientist                                                                                 |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| _Image classification (MNIST) using Convnets_                    | 📕 [Data Owner Notebook](./mnist/MNIST_Syft_Data_Owner.ipynb)                           | 📘 [Data Scientist Notebook](./mnist/MNIST_Syft_Data_Scientist.ipynb)                           |
| _Word level Language Modeling using LSTM RNNs_                   | Coming Soon                                                                            | Coming Soon                                                                                    |
| _Implement the Neural Style Transfer algorithm on images_        | Coming Soon                                                                            | Coming Soon                                                                                    |
| _Variational Auto-Encoders_                                      | 📕 [Data Owner Notebook](./vae/AutoEncoder_Syft_Data_Owner.ipynb)                       | 📘 [Data Scientist Notebook](./vae/AutoEncoder_Syft_Data_Scientist.ipynb)                       |
| _Training a CartPole to balance in OpenAI Gym with actor-critic_ | 📕 [Data Owner Notebook](./reinforcement_learning/Actor_Critic_Syft_Data_Owner.ipynb)   | 📘 [Data Scientist Notebook](./reinforcement_learning/Actor_Critic_Syft_Data_Scientist.ipynb)   |
| _Time sequence prediction - use an LSTM to learn Sine waves_     | 📕 [Data Owner Notebook](./time_sequence_prediction/TimeSequence_Syft_Data_Owner.ipynb) | 📘 [Data Scientist Notebook](./time_sequence_prediction/TimeSequence_Syft_Data_Scientist.ipynb) |
| _Superresolution using an efficient sub-pixel CNNs_              | 📕 [Data Owner Notebook](./super_resolution/SuperResolution_Syft_Data_Owner.ipynb)      | 📘 [Data Scientist Notebook](./super_resolution/SuperResolution_Syft_Data_Scientist.ipynb)      |
| _SNLI with GloVe vectors, LSTMs, and torchtext_                  | Coming Soon                                                                            | Coming Soon                                                                                    |
| _Generative Adversarial Networks (DCGAN)_                        | 📕 [Data Owner Notebook](./dcgan/DCGAN_Syft_Data_Owner.ipynb)                           | 📘 [Data Scientist Notebook](./dcgan/DCGAN_Syft_Data_Scientist.ipynb)                           |
