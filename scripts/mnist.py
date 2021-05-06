"""
Warm the MNIST cache: https://github.com/pytorch/vision/issues/1938
"""
# third party
from six.moves import urllib
from torchvision import datasets

# syft absolute
from syft.util import get_root_data_path

real_user_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36"
)
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", real_user_agent)]
urllib.request.install_opener(opener)

# https://github.com/pytorch/vision/issues/3549
datasets.MNIST.resources = [
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
]


datasets.MNIST(get_root_data_path(), train=True, download=True)
datasets.MNIST(get_root_data_path(), train=False, download=True)
