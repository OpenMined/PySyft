"""Which of two peers sends the offer."""

from syft.sync.connections.webrtc.real.aiortc_peer_connections import is_offerer

DO = "do@example.com"
# Emails compare in lower case. A case-sensitive comparison puts "DS" before "do".
DS = "DS@example.com"


def test_email_that_sorts_first_offers():
    assert is_offerer(DO, DS)
    assert not is_offerer(DS, DO)
