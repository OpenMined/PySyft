# third party
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import Layout
from ipywidgets import fixed
from ipywidgets import interact
from ipywidgets import interact_manual
from ipywidgets import interactive

# syft absolute
import syft as sy

duet = None

duetJoinButton = Button(
    description='Join Duet',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Join Duet',
    icon='fa-plug' # (FontAwesome names without the `fa-` prefix)
)

duetJoinButtonWithLoopback = Button(
    description='Join Duet(with loopback)',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Join Duet(with loopback)',
    layout=Layout(width='33%', height='80px'),
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

# view store , disconnect from duet, connect to grid
viewDuetStore = Button(
    description='View Duet Store',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='View Duet Store',
    icon='fa-archive' # (FontAwesome names without the `fa-` prefix)
)

disconnectDuetButton = Button(
    description='Disconnect from Duet',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Disconnect from Duet',
    icon='fa-window-close-o' # (FontAwesome names without the `fa-` prefix)
)

gridConnectButton = Button(
    description='Join Grid',
    disabled=False,
    layout=Layout(width='33%', height='80px'),
    button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Join Grid',
    icon='fa-plug' # (FontAwesome names without the `fa-` prefix)
)


dashboardDuetLogger = widgets.Output(layout={'border': '1px solid black'})

def on_join_duet_button(b):
    with dashboardDuetLogger:
        global duet
        duet = sy.join_duet()

def on_join_duet_loopback_button(b):
    with dashboardDuetLogger:
        global duet
        duet = sy.join_duet(loopback=True)

def view_store_button(b):
    with dashboardDuetLogger:
        global duet
        print(f"Duet Store {duet.store}")

duetJoinButton.on_click(on_join_duet_button)
duetJoinButtonWithLoopback.on_click(on_join_duet_loopback_button)
viewDuetStore.on_click(view_store_button)


line1 = widgets.HBox([duetJoinButton, duetJoinButtonWithLoopback, viewPeers])
line2 = widgets.HBox([viewDuetStore, disconnectDuetButton, gridConnectButton])
dashboard = widgets.VBox([line1,line2])