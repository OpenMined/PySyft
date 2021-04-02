# third party
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import fixed
from ipywidgets import interact
from ipywidgets import interact_manual
from ipywidgets import interactive
from loguru import logger

# syft absolute
import syft as sy

DUET = None

duetLaunchButton = widgets.Button(
    description='Launch Duet',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Launch Duet',
    layout=Layout(width='33%', height='80px'),
    icon='fa-plug' # (FontAwesome names without the `fa-` prefix)
)

duetLaunchButtonLoopback = widgets.Button(
    description='Launch Duet(with loopback)',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Launch Duet(with loopback)',
    icon='fa-plug' # (FontAwesome names without the `fa-` prefix)
)

viewPeers = widgets.Button(
    description='View Peers',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='View Peers',
    icon='fa-eye' # (FontAwesome names without the `fa-` prefix)
)

dashboardDuetLogger = widgets.Output(
    layout={
        "border": "1px solid black",
        "display": "flex",
        "flex_flow": "column",
        "align_items": "stretch",
    }
)

def on_launch_duet_button(b):
    dashboardDuetLogger.clear_output()
    with dashboardDuetLogger :
        DUET = sy.launch_duet()

def on_launch_duet_button_loopback(b):
    dashboardDuetLogger.clear_output()
    with dashboardDuetLogger:
        DUET = sy.launch_duet(loopback=True)

duetLaunchButton.on_click(on_launch_duet_button)
duetLaunchButtonLoopback.on_click(on_launch_duet_button_loopback)

dashboardLogger = widgets.Output(layout={'border': '1px solid black' , 'display' : 'flex' , 'flex_flow' : 'column' , 'align_items' : 'stretch' })
dashboard = widgets.HBox([duetLaunchButton, duetLaunchButtonLoopback, viewPeers])
