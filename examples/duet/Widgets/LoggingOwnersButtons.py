# stdlib
import re

# third party
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import Layout
from loguru import logger
import pandas as pd
import qgrid

# syft absolute
import syft as sy

LOG_LIST = []
ERROR_LIST = ['EXCEPTION', 'CRITICAL', 'ERROR' , 'WARNING' , 'INFO', 'DEBUG' , 'TRACE', 'NULL']
SEARCH_FILTER_LOG_WIDGET = widgets.Output(
    layout={
        "border": "1px solid black",
        "display": "flex",
        "flex_flow": "column",
        "align_items": "stretch",
    }
)
checkBoxDict = {}

def SET_LOGGING_SINK():
    logger.add(lambda msg: LOG_LIST.append(msg))

searchButton = Button(
    description='Search Logs',
    disabled=False,
    layout=Layout(width='20%', height='50px'),
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Search Logs',
    icon='fa-search' ,
)

filterLogButton = Button(
    description='Filter Logs',
    disabled=False,
    layout=Layout(width='20%', height='50px'),
    button_style='info',
    tooltip='Filter Logs',
    icon='fa-filter',
)

searchTextInput = widgets.Text(
    value='',
    placeholder='Enter Search String',
    description='Log Search',
    disabled=False
)

searchLogButton = widgets.Button(
    description='Search Logs',
    disabled=False,
    button_style='Success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Search',
    icon='fa-search' # (FontAwesome names without the `fa-` prefix)
)

for error in ERROR_LIST:
    checkBoxDict[error] = widgets.Checkbox(
        value=False,
        description=error,
        disabled=False,
        indent=False,
    )


def filterLog(b):
    selectedIDs, dtimeList, logTypeList, msgList = [] , [] , [] , []

    for i, error in enumerate(ERROR_LIST):
        if checkBoxDict[error].value :
            selectedIDs.append(i)

    for i, log in enumerate(LOG_LIST):
        for IDs in selectedIDs:
            if log.split('|')[1].strip() == ERROR_LIST[IDs]:
                dtime, logType, msg = log.split('|')
                dtimeList.append(dtime)
                logTypeList.append(logType)
                msgList.append(msg)

    SEARCH_FILTER_LOG_WIDGET.clear_output()
    with SEARCH_FILTER_LOG_WIDGET:
        dataDict = {
            "LOG TYPE" : logTypeList,
            "MESSAGE" : msgList,
        }
        df= pd.DataFrame(data=dataDict)
        qgrid_widget = qgrid.show_grid(df , show_toolbar = True)
        display(qgrid_widget)


def searchLogs(b):
    selectedIDs, dtimeList, logTypeList, msgList = [] , [] , [] , []

    for i, log in enumerate(LOG_LIST):
        dtime, logType, msg = log.split('|')
        if re.search(searchTextInput.value , msg):
            dtimeList.append(dtime)
            logTypeList.append(logType)
            msgList.append(msg)

    SEARCH_FILTER_LOG_WIDGET.clear_output()
    with SEARCH_FILTER_LOG_WIDGET:
        dataDict = {
            "LOG TYPE" : logTypeList,
            "MESSAGE" : msgList,
        }
        df= pd.DataFrame(data=dataDict)
        qgrid_widget = qgrid.show_grid(df , show_toolbar = True)
        display(qgrid_widget)



filterLogButton.on_click(filterLog)
searchLogButton.on_click(searchLogs)
topLineCheckbox = widgets.HBox([checkBoxDict[error] for error in ERROR_LIST[ : 4]])
bottomLineCheckbox = widgets.HBox([checkBoxDict[error] for error in ERROR_LIST[4 : ]])
CheckboxDashboard = widgets.VBox([topLineCheckbox, bottomLineCheckbox])
logFilterWidget = widgets.HBox([filterLogButton, CheckboxDashboard])
logSearchWidget = widgets.HBox([searchTextInput, searchLogButton])
FSEWidget = widgets.VBox([logFilterWidget , logSearchWidget])




