# third party
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import Layout
from loguru import logger

# syft absolute
import syft as sy

duet = None

LOG_LIST = []

def SET_LOGGING_SINK():
    logger.add(lambda msg: LOG_LIST.append(msg))

searchButton = Button(
    description='Search Logs',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Search Logs',
    icon='fa-search' ,
)

filterLogButton = Button(
    description='Filter Logs',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='info',
    tooltip='Filter Logs',
    icon='fa-filter',
)


