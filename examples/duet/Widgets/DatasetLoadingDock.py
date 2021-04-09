# stdlib
import re

# third party
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import Layout
from loguru import logger
import pandas as pd
import qgrid
from torchvision import datasets

# syft absolute
import syft as sy
from syft.util import get_root_data_path

LoadDisplayWidget = widgets.Output(layout={'border': '1px solid black' , 'display' : 'flex' , 'flex_flow' : 'column' , 'align_items' : 'stretch' })

def selectMNIST():
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
    print("Loaded MNIST Data Successfully .... ")

datasetSelectionWidget = widgets.Dropdown(
    options=["MNIST", "CIFAR", "COCO", "EMNIST" , "Flickr" , "Imagenet", "Custom"],
    value= "MNIST",
    layout= Layout(width='40%', height='50px'),
    description=" Dataset:",
    disabled=False,
)

datasetSelectionButton = Button(
    description='Load Dataset',
    disabled=False,
    layout=Layout(width='40%', height='60px'),
    button_style='info',
    tooltip='Load Dataset',
    icon='fa-data',
)

LibrarySelectionWidget = widgets.Dropdown(
    options=["tenseal" , "opacus", "sympc", "openmined_psi" ],
    value = "tenseal",
    layout= Layout(width='40%', height='60px'),
    description="Library: ",
    disabled=False,
)

librarySelectionButton = Button(
    description='Load Library',
    disabled=False,
    layout=Layout(width='40%', height='60px'),
    button_style='success',
    tooltip='Load Library',
    icon='fa-book',
)

def loadLibrary(b):
    LoadDisplayWidget.clear_output()
    with LoadDisplayWidget:
        if datasetSelectionWidget.value == "MNIST" :
            selectMNIST()

datasetSelectionButton.on_click(loadLibrary)
datasetWidget = widgets.HBox([datasetSelectionWidget, datasetSelectionButton])
libraryWidget = widgets.HBox([LibrarySelectionWidget, librarySelectionButton ])
loadWidget = widgets.VBox([datasetWidget, libraryWidget , LoadDisplayWidget ])
