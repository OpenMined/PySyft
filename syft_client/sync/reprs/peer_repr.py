from io import StringIO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from syft_client.sync.peers.peer import Peer, PeerState


def get_peer_list_table(peers: list[Peer]) -> str:
    console = Console(
        file=StringIO(),
        record=True,
        force_jupyter=False,
    )

    # Split by state
    approved_peers = [p for p in peers if p.state == PeerState.ACCEPTED]
    requested_by_me = [p for p in peers if p.state == PeerState.REQUESTED_BY_ME]
    requested_by_peer = [p for p in peers if p.state == PeerState.REQUESTED_BY_PEER]

    # Create main table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(width=None)

    # Section: Approved peers
    table.add_row("[bold green]client.peers[/]  [dim green]\\[0] or ['email'][/]")

    idx = 0
    for p in approved_peers:
        platform_modules = [plat.module_name for plat in p.platforms]
        table.add_row(
            f"  [dim black]\\[{idx}][/] [black]{p.email}[/]         [green]✓[/] [black]{', '.join(platform_modules)}[/]"
        )
        idx += 1

    # Add empty row for spacing
    table.add_row("")

    # Section: Requested by me (outgoing, waiting for peer to reciprocate)
    if requested_by_me:
        table.add_row(
            f"[bold blue]requested by me[/]  [dim blue]{len(requested_by_me)} pending[/]"
        )
        for p in requested_by_me:
            platform_modules = [plat.module_name for plat in p.platforms]
            table.add_row(
                f"  [dim black]\\[{idx}][/] [blue]{p.email}[/]         [blue]→[/] [black]{', '.join(platform_modules)}[/]"
            )
            idx += 1
    else:
        table.add_row("[bold blue]requested by me[/]  [dim blue]None[/]")

    table.add_row("")

    # Section: Requested by peer (incoming, waiting for us to approve)
    if requested_by_peer:
        table.add_row(
            f"[bold yellow]requested by peer[/]  [dim yellow]{len(requested_by_peer)} pending[/]"
        )
        for p in requested_by_peer:
            platform_modules = [plat.module_name for plat in p.platforms]
            table.add_row(
                f"  [dim black]\\[{idx}][/] [yellow]{p.email}[/]         [yellow]←[/] [black]{', '.join(platform_modules)}[/]"
            )
            idx += 1
    else:
        table.add_row("[bold yellow]requested by peer[/]  [dim yellow]None[/]")

    table.add_row("")

    # Wrap in panel with updated count
    panel = Panel(
        table,
        title=(
            f"Peers  ({len(approved_peers)} active, "
            f"{len(requested_by_me)} outgoing, "
            f"{len(requested_by_peer)} incoming)"
        ),
        expand=False,
        border_style="dim",
    )

    console.print(panel)
    return console.export_html(inline_styles=True)
