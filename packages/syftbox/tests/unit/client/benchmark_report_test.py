import json

import pytest

from syftbox.client.benchmark.report import ConsoleReport, JSONReport


class MockBenchmarkResult:
    """Mock class for benchmark results"""

    def dict_report(self):
        return {"metric1": 100, "metric2": 200}

    def readable_report(self):
        return "Metric1: 100\nMetric2: 200"


@pytest.fixture
def mock_metrics():
    """Fixture providing test metrics"""
    return {"test1": MockBenchmarkResult(), "test2": MockBenchmarkResult()}


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing temporary directory for output"""
    return tmp_path


def test_json_report_generate(mock_metrics, temp_output_dir):
    """Test JSON report generation"""
    reporter = JSONReport(temp_output_dir)
    reporter.generate(mock_metrics)

    # Verify file was created
    output_file = temp_output_dir / "benchmark_report.json"
    assert output_file.exists()

    # Verify file contents
    with open(output_file) as f:
        report_data = json.load(f)

    assert "result" in report_data
    assert len(report_data["result"]) == 2
    assert "test1" in report_data["result"]
    assert "test2" in report_data["result"]

    # Check content structure
    test1_data = report_data["result"]["test1"]
    assert test1_data["metric1"] == 100
    assert test1_data["metric2"] == 200


def test_console_report_generate(mock_metrics, capsys):
    """Test console report generation"""
    reporter = ConsoleReport()
    reporter.generate(mock_metrics)

    # Capture printed output
    captured = capsys.readouterr()

    # Verify output contains expected content
    assert "Metric1: 100" in captured.out
    assert "Metric2: 200" in captured.out
    assert "\n\n" in captured.out  # Check separator between reports


def test_json_report_file_error(mock_metrics, temp_output_dir, monkeypatch):
    """Test JSON report handling of file write errors"""

    def mock_open(*args, **kwargs):
        raise IOError("Mock file write error")

    monkeypatch.setattr("builtins.open", mock_open)

    reporter = JSONReport(temp_output_dir)
    with pytest.raises(IOError):
        reporter.generate(mock_metrics)


def test_json_report_with_empty_metrics(temp_output_dir):
    """Test JSON report generation with empty metrics"""
    reporter = JSONReport(temp_output_dir)
    reporter.generate({})

    output_file = temp_output_dir / "benchmark_report.json"
    with open(output_file) as f:
        report_data = json.load(f)

    assert report_data["result"] == {}


def test_console_report_with_empty_metrics(capsys):
    """Test console report generation with empty metrics"""
    reporter = ConsoleReport()
    reporter.generate({})

    captured = capsys.readouterr()
    assert captured.out.strip() == ""


def test_json_report_output_format(mock_metrics, temp_output_dir):
    """Test JSON report output formatting"""
    reporter = JSONReport(temp_output_dir)
    reporter.generate(mock_metrics)

    output_file = temp_output_dir / "benchmark_report.json"
    with open(output_file) as f:
        content = f.read()

    # Verify JSON is properly indented
    assert "    " in content  # Check for indentation
    assert "}" in content  # Check for proper JSON structure

    # Verify it's valid JSON by parsing it
    assert json.loads(content) is not None


def test_console_report_multiple_metrics(capsys):
    """Test console report with multiple different metrics"""

    class CustomMockResult:
        def readable_report(self):
            return "Custom Report"

    metrics = {"test1": MockBenchmarkResult(), "test2": CustomMockResult()}

    reporter = ConsoleReport()
    reporter.generate(metrics)

    captured = capsys.readouterr()
    assert "Metric1: 100" in captured.out
    assert "Custom Report" in captured.out
