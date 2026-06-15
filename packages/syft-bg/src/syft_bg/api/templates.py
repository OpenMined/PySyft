"""Display templates for API result objects."""

STATUS_TEMPLATE = """\
syft-bg status
{sep}
  email:       {email}
  syftbox:     {syftbox_root}
  environment: {env}

tokens
{line}
{tokens}

services
{line}
{services}"""

AUTO_APPROVALS_SECTION = """

auto-approval objects
{line}
{contents}"""

APPROVED_DOMAINS_SECTION = """

auto-approved domains
{line}
{contents}"""
