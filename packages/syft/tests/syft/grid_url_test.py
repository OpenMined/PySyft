# third party
import pytest

# syft absolute
from syft.types.grid_url import GridURL

test_suite = [
    ("http://0.0.0.0", 8081, "http://0.0.0.0:8081"),
    ("http://0.0.0.0", None, "http://0.0.0.0:80"),
    (None, None, "http://localhost:80"),
    ("http://0.0.0.0:8081", 8082, "http://0.0.0.0:8081"),
    ("0.0.0.0:8081", None, "http://0.0.0.0:8081"),
    ("domainname.com", None, "http://domainname.com:80"),
    ("https://domainname.com", None, "https://domainname.com:80"),
]


@pytest.mark.parametrize("url, port, ground_truth", test_suite)
def test_grid_url(url, port, ground_truth) -> None:
    if not url and not port:
        assert GridURL().base_url == ground_truth
    elif not url:
        assert GridURL(port=port).base_url == ground_truth
    elif not port:
        assert GridURL(host_or_ip=url).base_url == ground_truth
    else:
        assert GridURL(host_or_ip=url, port=port).base_url == ground_truth
