"""Category 0: Parsing and validation of permission YAML files."""

import pytest

from syft_permissions.spec.access import Access
from syft_permissions.spec.rule import Rule
from syft_permissions.spec.ruleset import RuleSet


def test_load_valid_yaml(tmp_path):
    yaml_content = """\
rules:
  - pattern: "**"
    access:
      read:
        - "*"
      write:
        - "owner@test.com"
terminal: false
"""
    path = tmp_path / "syft.pub.yaml"
    path.write_text(yaml_content)

    rs = RuleSet.load(path)
    assert len(rs.rules) == 1
    assert rs.rules[0].pattern == "**"
    assert rs.rules[0].access.read == ["*"]
    assert rs.rules[0].access.write == ["owner@test.com"]
    assert rs.terminal is False
    assert rs.path == str(tmp_path)


def test_load_valid_yaml_with_terminal(tmp_path):
    yaml_content = """\
rules:
  - pattern: "**"
    access:
      read:
        - "*"
terminal: true
"""
    path = tmp_path / "syft.pub.yaml"
    path.write_text(yaml_content)

    rs = RuleSet.load(path)
    assert len(rs.rules) == 1
    assert rs.terminal is True


def test_missing_fields_default():
    rs = RuleSet.model_validate({})
    assert rs.rules == []
    assert rs.terminal is False


def test_missing_access_fields_default():
    access = Access.model_validate({})
    assert access.admin == []
    assert access.write == []
    assert access.read == []


def test_unsupported_template_userhash_raises():
    with pytest.raises(ValueError, match="Unsupported template"):
        Rule(pattern="{{.UserHash}}/**", access=Access())


def test_unsupported_template_year_raises():
    with pytest.raises(ValueError, match="Unsupported template"):
        Rule(pattern="reports/{{.Year}}/**", access=Access())


def test_unsupported_template_month_raises():
    with pytest.raises(ValueError, match="Unsupported template"):
        Rule(pattern="reports/{{.Month}}/**", access=Access())


def test_unsupported_template_date_raises():
    with pytest.raises(ValueError, match="Unsupported template"):
        Rule(pattern="reports/{{.Date}}/**", access=Access())


def test_supported_template_useremail_ok():
    rule = Rule(pattern="{{.UserEmail}}/**", access=Access(read=["USER"]))
    assert "{{.UserEmail}}" in rule.pattern


def test_save_and_reload(tmp_path):
    rs = RuleSet(
        rules=[Rule(pattern="**", access=Access(read=["*"]))],
        terminal=True,
        path=str(tmp_path),
    )
    rs.save()
    loaded = RuleSet.load(tmp_path / "syft.pub.yaml")
    assert len(loaded.rules) == 1
    assert loaded.terminal is True


def test_path_excluded_from_serialization():
    rs = RuleSet(path="/some/path", rules=[])
    dumped = rs.model_dump(mode="json")
    assert "path" not in dumped
