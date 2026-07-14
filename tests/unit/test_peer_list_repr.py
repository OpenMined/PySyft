"""Tests for PeerList's human-friendly summary rendering (__str__/_repr_html_/__repr__)."""

import pytest
from syft_client.sync.peers.peer import Peer, PeerState
from syft_client.sync.peers.peer_list import PeerList


def _mixed_peer_list() -> PeerList:
    """A PeerList with one peer in each visible state, in required sort order."""
    return PeerList(
        [
            Peer(email="do@org.com", state=PeerState.ACCEPTED),
            Peer(email="other@org.com", state=PeerState.REQUESTED_BY_ME),
            Peer(email="incoming@org.com", state=PeerState.REQUESTED_BY_PEER),
        ]
    )


def test_str_header_shows_count():
    pl = _mixed_peer_list()
    assert str(pl).startswith("👥 Your peers (3):")


def test_str_renders_emoji_and_friendly_labels():
    text = str(_mixed_peer_list())
    assert "✅ do@org.com" in text
    assert "— connected" in text
    assert "⏳ other@org.com" in text
    assert "requested_by_us (waiting for approval)" in text
    assert "📩 incoming@org.com" in text
    assert "requested_by_peer (waiting for your approval)" in text


def test_connected_tip_present_when_connected_peer_exists():
    text = str(_mixed_peer_list())
    assert "client.datasets.get" in text
    assert 'datasite="do@org.com"' in text


def test_connected_tip_absent_when_no_connected_peer():
    pl = PeerList(
        [
            Peer(email="other@org.com", state=PeerState.REQUESTED_BY_ME),
            Peer(email="incoming@org.com", state=PeerState.REQUESTED_BY_PEER),
        ]
    )
    assert "client.datasets.get" not in str(pl)


def test_pending_tip_present_when_pending_peer_exists():
    text = str(_mixed_peer_list())
    assert "pending requests" in text


def test_pending_tip_absent_when_only_connected():
    pl = PeerList([Peer(email="do@org.com", state=PeerState.ACCEPTED)])
    assert "pending requests" not in str(pl)


def test_empty_peer_list_message():
    assert str(PeerList([])) == "👥 You have no peers yet."


def test_repr_html_wraps_and_escapes_summary():
    html = _mixed_peer_list()._repr_html_()
    assert html is not None
    assert html.startswith("<pre>")
    assert html.endswith("</pre>")
    assert "👥 Your peers (3):" in html


def test_repr_html_escapes_html_in_email():
    pl = PeerList([Peer(email="a<b>@org.com", state=PeerState.ACCEPTED)])
    html = pl._repr_html_()
    assert html is not None
    assert "a<b>@org.com" not in html
    assert "a&lt;b&gt;@org.com" in html


def test_repr_html_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("SYFT_NO_REPR_HTML", "1")
    assert _mixed_peer_list()._repr_html_() is None


def test_repr_preserves_technical_string():
    assert repr(_mixed_peer_list()).startswith("PeerList(")


def test_peer_list_len_and_iter():
    pl = _mixed_peer_list()
    assert len(pl) == 3
    assert [p.email for p in pl] == [
        "do@org.com",
        "other@org.com",
        "incoming@org.com",
    ]


def test_peer_list_getitem_int():
    assert _mixed_peer_list()[0].email == "do@org.com"


def test_peer_list_getitem_str():
    assert _mixed_peer_list()["other@org.com"].state == PeerState.REQUESTED_BY_ME


def test_peer_list_getitem_str_not_found():
    with pytest.raises(ValueError, match="not found"):
        _mixed_peer_list()["nobody@org.com"]


def test_peer_list_getitem_invalid_type():
    with pytest.raises(TypeError, match="Invalid index type"):
        _mixed_peer_list()[3.14]  # type: ignore[index]


def test_peer_list_rejects_non_peer_items():
    with pytest.raises(TypeError):
        PeerList(["not-a-peer"])  # type: ignore[list-item]


def test_peer_list_sorts_unsorted_input():
    # Construction sorts by state (accepted → requested_by_me → requested_by_peer)
    # rather than requiring the caller to pre-sort.
    pl = PeerList(
        [
            Peer(email="incoming@org.com", state=PeerState.REQUESTED_BY_PEER),
            Peer(email="connected@org.com", state=PeerState.ACCEPTED),
            Peer(email="outgoing@org.com", state=PeerState.REQUESTED_BY_ME),
        ]
    )
    assert [p.state for p in pl] == [
        PeerState.ACCEPTED,
        PeerState.REQUESTED_BY_ME,
        PeerState.REQUESTED_BY_PEER,
    ]


def test_peer_list_sort_is_stable_within_a_state():
    # Peers sharing a state keep their input order (stable sort).
    pl = PeerList(
        [
            Peer(email="second@org.com", state=PeerState.ACCEPTED),
            Peer(email="first@org.com", state=PeerState.ACCEPTED),
        ]
    )
    assert [p.email for p in pl] == ["second@org.com", "first@org.com"]
