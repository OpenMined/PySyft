# third party
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import Layout
from ipywidgets import fixed
from ipywidgets import interact
from ipywidgets import interact_manual
from ipywidgets import interactive
from loguru import logger

# syft absolute
import syft as sy

DUET = None

# Join Duet
duetJoinButton = Button(
    description='Join Duet',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Join Duet',
    icon='fa-plug' # (FontAwesome names without the `fa-` prefix)
)

# Joining Duet With Loopback is True
duetJoinButtonWithLoopback = Button(
    description='Join Duet(with loopback)',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Join Duet(with loopback)',
    layout=Layout(width='33%', height='80px'),
    icon='fa-plug' # (FontAwesome names without the `fa-` prefix)
)

# View all Peers
viewPeers = widgets.Button(
    description='View Peers',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='View Peers',
    icon='fa-eye' # (FontAwesome names without the `fa-` prefix)
)

# view store , disconnect from duet, connect to grid
viewDuetStore = Button(
    description='View Duet Store',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='View Duet Store',
    icon='fa-archive' # (FontAwesome names without the `fa-` prefix)
)

# Disconnect from Duet and shutdown
disconnectDuetButton = Button(
    description='Disconnect from Duet',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Disconnect from Duet',
    icon='fa-window-close-o' # (FontAwesome names without the `fa-` prefix)
)

# Button to support Openmined
supportOpenMinedButton = Button(
    description='Support OpenMined',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Support',
    icon='fa-money' # (FontAwesome names without the `fa-` prefix)
)
def on_join_duet_button(b):
    dashboardDuetLogger.clear_output()
    with dashboardDuetLogger:
        DUET = sy.join_duet()

def on_join_duet_loopback_button(b):
    dashboardDuetLogger.clear_output()
    with dashboardDuetLogger:
        DUET = sy.join_duet(loopback=True)

def view_store_button(b):
    storeLogger.clear_output()
    with storeLogger:
        global duet
        qgrid_widget = qgrid.show_grid(duet.store.pandas , show_toolbar = True)
        display(qgrid_widget)

def view_peers(b):
    dashboardDuetLogger.clear_output()
    with dashboardDuetLogger:
        print(f"Peers {sy.grid.duet.get_available_network()}")


def support_openmined(b):
    dashboardDuetLogger.clear_output()
    with dashboardDuetLogger:
        print(sy.grid.duet.generate_donation_msg(name="Openmined"))

duetJoinButton.on_click(on_join_duet_button)
duetJoinButtonWithLoopback.on_click(on_join_duet_loopback_button)
viewDuetStore.on_click(view_store_button)
viewPeers.on_click(view_peers)
supportOpenMinedButton.on_click(support_openmined)

dashboardDuetLogger = widgets.Output(layout={'border': '1px solid black'})
storeLogger = widgets.Output(layout={'border': '1px solid black'})

line1 = widgets.HBox([duetJoinButton, duetJoinButtonWithLoopback, viewPeers])
line2 = widgets.HBox([viewDuetStore, disconnectDuetButton, supportOpenMinedButton])
dashboard = widgets.VBox([line1,line2])
