import pytest

from syft_client.gdrive_utils import _extract_auth_code


def test_raw_code_returned_as_is():
    code = "4/0AFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEAA"
    assert _extract_auth_code(code) == code


def test_full_redirect_url_extracts_code():
    fake_code = "4/0AFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEcodeFAKEAA"
    url = (
        "http://localhost:1/?state=FAKEstateFAKEstateFAKEstateFA"
        "&iss=https://accounts.google.com"
        f"&code={fake_code}"
        "&scope=https://www.googleapis.com/auth/drive"
    )
    assert _extract_auth_code(url) == fake_code


def test_https_url_extracts_code():
    url = "https://example.com/callback?code=abc123&state=xyz"
    assert _extract_auth_code(url) == "abc123"


def test_url_without_code_raises():
    with pytest.raises(ValueError, match="Could not find 'code'"):
        _extract_auth_code("http://localhost:1/?state=xyz")
