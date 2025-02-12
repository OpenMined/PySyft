from syftbox.lib.debug import debug_report, debug_report_yaml


def test_debug_report(mock_config):
    result = debug_report()
    assert isinstance(result, dict)
    assert "system" in result
    assert "syftbox" in result
    assert "syftbox_env" in result
    assert "resources" in result["system"]
    assert "os" in result["system"]
    assert "python" in result["system"]
    assert "command" in result["syftbox"]
    assert "client_config_path" in result["syftbox"]
    assert "client_config" in result["syftbox"]
    assert "apps_dir" in result["syftbox"]
    assert "apps" in result["syftbox"]


def test_debug_report_readable(mock_config):
    result = debug_report_yaml()
    assert isinstance(result, str)
    assert "system" in result
    assert "syftbox" in result
    assert "syftbox_env" in result
    assert "resources" in result
    assert "os" in result
    assert "python" in result
    assert "command" in result
    assert "client_config_path" in result
    assert "client_config" in result
    assert "apps_dir" in result
    assert "apps" in result
