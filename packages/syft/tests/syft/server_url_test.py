# third party
import pytest

# syft absolute
from syft.types.server_url import ServerURL

test_suite = [
    ("http://0.0.0.0", 8081, "http://0.0.0.0:8081"),
    ("http://0.0.0.0", None, "http://0.0.0.0:80"),
    (None, None, "http://localhost:80"),
    ("http://0.0.0.0:8081", 8082, "http://0.0.0.0:8081"),
    ("0.0.0.0:8081", None, "http://0.0.0.0:8081"),
    ("example.com", None, "http://example.com:80"),
    ("https://example.com", None, "https://example.com:80"),
]


@pytest.mark.parametrize("url, port, ground_truth", test_suite)
def test_server_url(url, port, ground_truth) -> None:
    if not url and not port:
        assert ServerURL().base_url == ground_truth
    elif not url:
        assert ServerURL(port=port).base_url == ground_truth
    elif not port:
        assert ServerURL(host_or_ip=url).base_url == ground_truth
    else:
        assert ServerURL(host_or_ip=url, port=port).base_url == ground_truth
