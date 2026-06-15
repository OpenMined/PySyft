"""Result dataclasses returned by syft-bg API functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, computed_field

from syft_bg.common.syft_bg_config import SyftBgConfig
from syft_bg.services.base import ServiceInfo, ServiceStatus
from typing import Optional

if TYPE_CHECKING:
    from syft_bg.approve.config import AutoApprovalObj


class InitResult(BaseModel):
    """Result of init() call."""

    success: bool
    config_path: Path | None = None
    error: str | None = None
    services: dict[str, tuple[bool, str]] | None = None
    issues: list[str] = Field(default_factory=list)

    def __repr__(self) -> str:
        lines = []
        if self.success:
            lines.append("InitResult: OK")
            if self.config_path:
                lines.append(f"  config: {self.config_path}")
            if self.services:
                for name, (ok, msg) in self.services.items():
                    status = "started" if ok else "failed"
                    lines.append(f"  {name}: {status} — {msg}")
        else:
            lines.append("InitResult: FAILED")
            if self.error:
                lines.append(f"  error: {self.error}")
        if self.issues:
            lines.append("  issues:")
            for issue in self.issues:
                indented = issue.replace("\n", "\n      ")
                lines.append(f"    - {indented}")
        return "\n".join(lines)


class AuthResult(BaseModel):
    """Result of authenticate() call."""

    success: bool
    gmail_ok: bool = False
    drive_ok: bool = False
    error: str | None = None

    def __repr__(self) -> str:
        lines = []
        if self.success:
            lines.append("AuthResult: OK")
        else:
            lines.append("AuthResult: FAILED")
            if self.error:
                lines.append(f"  error: {self.error}")
        lines.append(f"  gmail: {'ready' if self.gmail_ok else 'missing'}")
        lines.append(f"  drive: {'ready' if self.drive_ok else 'missing'}")
        return "\n".join(lines)


class TokenStatus(BaseModel):
    """Displayable token status with text and HTML representations."""

    label: str
    path: Optional[Path]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return self.path is not None and self.path.exists()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> str:
        if self.path is None:
            return "not configured"
        if self.path.exists():
            return "ready"
        return f"missing ({self.path})"

    def render(self, as_html: bool = False) -> str:
        if as_html:
            if self.ok:
                status = f'<span style="color:green">{self.status}</span>'
            elif self.path is not None:
                status = f'<span style="color:red">missing</span> ({self.path})'
            else:
                status = f'<span style="color:red">{self.status}</span>'
            return f"{self.label}: {status}"
        return f"  {self.label:<35} {self.status}"


class ServiceSatusLineRepr(BaseModel):
    """Displayable service status with text and HTML representations."""

    info: ServiceInfo

    def _installed_suffix(self, as_html: bool = False) -> str:
        if self.info.installed:
            if as_html:
                return ' <span style="color:green">(installed)</span>'
            return " (installed)"
        if as_html:
            return ' <span style="color:gray">(not installed)</span>'
        return " (not installed)"

    def _status_icon(self) -> str:
        if self.info.status == ServiceStatus.ERROR:
            return "🔴"
        if self.info.status == ServiceStatus.STARTING:
            return "🟡"
        if self.info.status == ServiceStatus.RUNNING:
            return "🟢"
        if self.info.status == ServiceStatus.STOPPED:
            return "⚪"
        if self.info.status == ServiceStatus.UNKNOWN:
            return "?"

    def render(self, as_html: bool = False) -> str:
        name = self.info.name
        suffix = self._installed_suffix(as_html)
        icon = self._status_icon()
        status_str = self.info.status_str
        if self.info.status == ServiceStatus.ERROR:
            if as_html:
                return (
                    f'{name}: <span style="color:red">{icon} setup error</span>{suffix}'
                )
            return f"  {name:<16} {icon} setup error{suffix}"
        if as_html:
            if self.info.status == ServiceStatus.STARTING:
                color = "orange"
            elif self.info.status == ServiceStatus.RUNNING:
                color = "green"
            else:
                color = "red"
            status = f'<span style="color:{color}">{icon} {status_str}</span>'
            return f"{name}: {status}{suffix}"
        return f"  {name:<16} {icon} {status_str}{suffix}"


class StatusResult(BaseModel):
    """Status of syft-bg services and configuration."""

    model_config = {"arbitrary_types_allowed": True}

    config: "SyftBgConfig"
    service_infos: dict[str, ServiceInfo] = Field(default_factory=dict)
    is_colab: bool = False

    @property
    def email(self) -> str | None:
        return self.config.do_email

    @property
    def syftbox_root(self) -> str | None:
        return self.config.syftbox_root

    @property
    def auto_approvals(self) -> dict[str, "AutoApprovalObj"]:
        return self.config.approve.auto_approvals.objects

    @property
    def approved_domains(self) -> list[str]:
        return self.config.approve.peers.approved_domains

    def _token_items(self) -> list[TokenStatus]:
        active = self.service_infos
        items: list[TokenStatus] = []

        if "notify" in active:
            n = self.config.notify
            items.append(
                TokenStatus(label="notify.gmail_token_path", path=n.gmail_token_path)
            )
            items.append(
                TokenStatus(label="notify.drive_token_path", path=n.drive_token_path)
            )
        if "approve" in active:
            a = self.config.approve
            items.append(
                TokenStatus(label="approve.drive_token_path", path=a.drive_token_path)
            )
        if "email_approve" in active:
            e = self.config.email_approve
            items.append(
                TokenStatus(
                    label="email_approve.gmail_token_path", path=e.gmail_token_path
                )
            )

        return items

    def _service_repr_lines(self) -> list[ServiceSatusLineRepr]:
        return [ServiceSatusLineRepr(info=info) for info in self.service_infos.values()]

    def _tokens_contents(self, as_html: bool = False) -> str:
        items = self._token_items()
        if not items:
            return "  (no active services)"
        sep = "<br>" if as_html else "\n"
        return sep.join(t.render(as_html) for t in items)

    def _services_contents(self, as_html: bool = False) -> str:
        sep = "<br>" if as_html else "\n"
        return sep.join(s.render(as_html) for s in self._service_repr_lines())

    def _auto_approval_obj_contents(self, name: str, obj: AutoApprovalObj) -> str:
        contents = "\n".join(
            f"    content: {e.relative_path}" for e in obj.file_contents
        )
        files = "\n".join(f"    file:   {f}" for f in obj.file_paths)
        peers = (
            f"    peers:  {', '.join(obj.peers)}" if obj.peers else "    peers:  (any)"
        )
        body = "\n".join(part for part in [contents, files, peers] if part)
        return f"  [{name}]\n{body}"

    def _auto_approvals_contents(self) -> str:
        return "\n".join(
            self._auto_approval_obj_contents(name, obj)
            for name, obj in self.auto_approvals.items()
        )

    def _approved_domains_contents(self) -> str:
        return "\n".join(f"  {d}" for d in self.approved_domains)

    def render(self, as_html: bool = False) -> str:
        if as_html:
            return self._repr_html_()
        return self.__repr__()

    def __repr__(self) -> str:
        from syft_bg.api.templates import (
            APPROVED_DOMAINS_SECTION,
            AUTO_APPROVALS_SECTION,
            STATUS_TEMPLATE,
        )

        line = "-" * 40
        result = STATUS_TEMPLATE.format(
            sep="=" * 40,
            email=self.email or "not configured",
            syftbox_root=self.syftbox_root or "not configured",
            env="Colab" if self.is_colab else "local",
            line=line,
            tokens=self._tokens_contents(),
            services=self._services_contents(),
        )

        if self.auto_approvals:
            result += AUTO_APPROVALS_SECTION.format(
                line=line, contents=self._auto_approvals_contents()
            )

        if self.approved_domains:
            result += APPROVED_DOMAINS_SECTION.format(
                line=line, contents=self._approved_domains_contents()
            )

        return result

    def _repr_html_(self) -> str:
        env = "Colab" if self.is_colab else "local"
        tokens_html = self._tokens_contents(as_html=True)
        services_html = self._services_contents(as_html=True)

        html = "<b>syft-bg status</b><br>"
        html += f"email: {self.email or 'not configured'}<br>"
        html += f"syftbox: {self.syftbox_root or 'not configured'}<br>"
        html += f"environment: {env}<br><br>"
        html += f"<b>tokens</b><br>{tokens_html}<br><br>"
        html += f"<b>services</b><br>{services_html}"
        return html


class InstallationResult(BaseModel):
    """Result of installing or uninstalling a single service."""

    success: bool
    service: str
    message: str = ""

    def __repr__(self) -> str:
        status = "ok" if self.success else "failed"
        return f"InstallationResult: {self.service} {status} — {self.message}"


class AutoApproveResult(BaseModel):
    """Result of auto_approve() call."""

    success: bool
    name: str = ""
    file_contents: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    peers: list[str] = Field(default_factory=list)
    error: str | None = None

    def __repr__(self) -> str:
        lines = []
        if self.success:
            lines.append(f"AutoApproveResult: OK [{self.name}]")
            if self.file_contents:
                lines.append("  file_contents:")
                for s in self.file_contents:
                    lines.append(f"    - {s}")
            if self.file_paths:
                lines.append(f"  file_paths: {', '.join(self.file_paths)}")
            if self.peers:
                lines.append(f"  peers: {', '.join(self.peers)}")
            else:
                lines.append("  peers: (any)")
        else:
            lines.append("AutoApproveResult: FAILED")
            if self.error:
                lines.append(f"  error: {self.error}")
        return "\n".join(lines)
