from types import SimpleNamespace

import pytest

from syftbox.app.install import check_os_compatibility


def test_os_compatibility_compatible():
    app_config_mock = SimpleNamespace(
        **{
            "app": SimpleNamespace(
                **{
                    "platforms": ["darwin", "linux"],
                }
            ),
        }
    )

    check_os_compatibility(app_config_mock)


def test_os_compatibility_incompatible():
    app_config_mock = SimpleNamespace(
        **{
            "app": SimpleNamespace(
                **{
                    "platforms": ["different_os"],
                }
            ),
        }
    )
    with pytest.raises(OSError) as e:
        check_os_compatibility(app_config_mock)
        assert e.value == "Your OS isn't supported by this app."


def test_os_compatibility_without_config():
    app_config_mock = SimpleNamespace(**{"app": {}})

    check_os_compatibility(app_config_mock)
