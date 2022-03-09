"""
Warm the MNIST cache: https://github.com/pytorch/vision/issues/1938
"""
# third party
from packaging import version
import torchvision

# syft absolute
from syft.util import get_root_data_path

# https://github.com/pytorch/vision/issues/3549
TORCHVISION_VERSION = version.parse(torchvision.__version__)

# Use our own GitHub Mirror because the torchvision ones keep 503ing randomly
URL = "https://raw.githubusercontent.com/madhavajay/datasets/main/MNIST/raw/"
if TORCHVISION_VERSION < version.parse("0.9.1"):
    torchvision.datasets.MNIST.resources = [
        (f"{URL}train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        (f"{URL}train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        (f"{URL}t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        (f"{URL}t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]
else:
    torchvision.datasets.MNIST.mirrors.insert(0, URL)

torchvision.datasets.MNIST(get_root_data_path(), train=True, download=True)
torchvision.datasets.MNIST(get_root_data_path(), train=False, download=True)
