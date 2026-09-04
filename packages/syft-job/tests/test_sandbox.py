"""Tests for the job sandbox.

The behavioural tests shell out to sandbox.py and inspect what the sandboxed
process can and cannot do. They deliberately assert on the *kernel's* behaviour
rather than on our code paths -- the guarantee is only worth what the kernel
actually enforces.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from syft_job import job_runner, sandbox

SANDBOX_PY = str(Path(sandbox.__file__).resolve())
LINUX_X86 = sys.platform == "linux" and os.uname().machine == "x86_64"
requires_linux_x86 = pytest.mark.skipif(
    not LINUX_X86, reason="sandbox is Linux/x86-64 only"
)


def _run_sandboxed(snippet: str, extra_args: list[str] | None = None):
    """Run a Python snippet under the sandbox, as the current user."""
    args = [
        sys.executable,
        SANDBOX_PY,
        "--uid",
        str(os.getuid()),
        "--gid",
        str(os.getgid()),
        *(extra_args or []),
        "--",
        sys.executable,
        "-c",
        snippet,
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=60)


# --------------------------------------------------------------------------
# configuration / wiring
# --------------------------------------------------------------------------


def test_sandbox_defaults_to_off(monkeypatch):
    """Must default off: this runner also runs on data owners' own machines."""
    monkeypatch.delenv(job_runner.SANDBOX_ENV_VAR, raising=False)
    assert job_runner.get_sandbox_mode() == "off"


def test_invalid_mode_rejected(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "sort-of")
    with pytest.raises(ValueError):
        job_runner.get_sandbox_mode()


def test_command_unwrapped_when_off(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "off")
    assert job_runner.build_job_command(Path("/tmp/run.sh")) == [
        "bash",
        "/tmp/run.sh",
    ]


@requires_linux_x86
@pytest.mark.parametrize("mode", ["on", "require"])
def test_command_wrapped_when_enabled(monkeypatch, mode):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, mode)
    cmd = job_runner.build_job_command(Path("/tmp/run.sh"))
    assert cmd[0] == sys.executable
    assert cmd[1] == SANDBOX_PY
    # the job itself is still the tail of the command
    assert cmd[-2:] == ["bash", "/tmp/run.sh"]


def test_require_raises_when_unsupported(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "require")
    monkeypatch.setattr(sandbox, "is_supported", lambda: (False, "no kernel support"))
    with pytest.raises(job_runner.SandboxUnavailableError):
        job_runner.build_job_command(Path("/tmp/run.sh"))


def test_on_degrades_when_unsupported(monkeypatch, capsys):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "on")
    monkeypatch.setattr(sandbox, "is_supported", lambda: (False, "no kernel support"))
    assert job_runner.build_job_command(Path("/tmp/run.sh")) == [
        "bash",
        "/tmp/run.sh",
    ]
    assert "WITHOUT network isolation" in capsys.readouterr().out


def test_env_allowlisted_when_sandboxed(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "on")
    monkeypatch.setenv("SYFT_BOOTSTRAP_SA_SECRET", "super-secret")
    monkeypatch.setenv("SYFT_ENCLAVE_TOKEN_CONTENT", "oauth-token")
    env = job_runner.build_job_env("/syftbox", "enclave@example.org")
    assert "SYFT_BOOTSTRAP_SA_SECRET" not in env
    assert "SYFT_ENCLAVE_TOKEN_CONTENT" not in env
    assert env["SYFTBOX_FOLDER"] == "/syftbox"
    assert env[job_runner.IS_IN_JOB_ENV_VAR] == "true"


def test_env_inherited_when_off(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "off")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")
    env = job_runner.build_job_env("/syftbox", "a@b.c")
    assert env["SOME_UNRELATED_VAR"] == "kept"


# --------------------------------------------------------------------------
# what the kernel actually enforces
# --------------------------------------------------------------------------


@requires_linux_x86
def test_supported_on_this_platform():
    supported, reason = sandbox.is_supported()
    assert supported, reason


@requires_linux_x86
@pytest.mark.parametrize(
    "family,name", [(2, "AF_INET"), (10, "AF_INET6"), (1, "AF_UNIX")]
)
def test_socket_creation_blocked(family, name):
    """Every address family, not just the internet ones.

    AF_UNIX matters: a socket's network namespace is fixed at creation, so a
    process that can open a unix socket can be handed an already-connected
    network socket over SCM_RIGHTS by a co-resident helper.
    """
    proc = _run_sandboxed(
        f"import socket\n"
        f"try:\n"
        f"    socket.socket({family}, socket.SOCK_STREAM)\n"
        f"    print('ALLOWED')\n"
        f"except OSError as e:\n"
        f"    print('BLOCKED')\n"
    )
    assert proc.stdout.strip() == "BLOCKED", f"{name} was not blocked: {proc.stdout}"


@requires_linux_x86
def test_restriction_survives_exec():
    """The guarantee is about compiled binaries, so it must outlive execve."""
    inner = (
        "import socket\n"
        "try:\n"
        "    socket.socket(2, 1); print('ALLOWED')\n"
        "except OSError: print('BLOCKED')\n"
    )
    proc = _run_sandboxed(
        "import subprocess, sys\n"
        f"r = subprocess.run([sys.executable, '-c', {inner!r}], "
        "capture_output=True, text=True)\n"
        "print(r.stdout.strip())\n"
    )
    assert proc.stdout.strip() == "BLOCKED"


@requires_linux_x86
def test_dns_resolution_blocked():
    proc = _run_sandboxed(
        "import socket\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 80); print('RESOLVED')\n"
        "except Exception: print('BLOCKED')\n"
    )
    assert proc.stdout.strip() == "BLOCKED"


@requires_linux_x86
def test_ordinary_work_unaffected():
    """A lockdown that breaks normal jobs is not shippable."""
    proc = _run_sandboxed(
        "import os, tempfile\n"
        "total = sum(i * i for i in range(10000))\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    p = os.path.join(d, 'out.txt')\n"
        "    open(p, 'w').write('result')\n"
        "    assert open(p).read() == 'result'\n"
        "print('OK', total)\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("OK 333283335000")


@requires_linux_x86
def test_no_new_privs_is_set():
    proc = _run_sandboxed(
        "print([l for l in open('/proc/self/status') if 'NoNewPrivs' in l][0].strip())"
    )
    assert proc.stdout.strip().endswith("1")


# --------------------------------------------------------------------------
# fail-closed
# --------------------------------------------------------------------------


@requires_linux_x86
def test_refuses_unknown_option_without_running_command():
    proc = subprocess.run(
        [sys.executable, SANDBOX_PY, "--nope", "--", "echo", "SHOULD_NOT_RUN"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == sandbox.REFUSED_EXIT_CODE
    assert sandbox.SENTINEL in proc.stderr
    assert "SHOULD_NOT_RUN" not in proc.stdout


@requires_linux_x86
def test_refuses_when_no_command_given():
    proc = subprocess.run(
        [sys.executable, SANDBOX_PY, "--"], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == sandbox.REFUSED_EXIT_CODE


@requires_linux_x86
def test_refuses_when_exec_target_missing():
    proc = _run_sandboxed_missing = subprocess.run(
        [
            sys.executable,
            SANDBOX_PY,
            "--uid",
            str(os.getuid()),
            "--gid",
            str(os.getgid()),
            "--",
            "/nonexistent/binary",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == sandbox.REFUSED_EXIT_CODE
    assert sandbox.SENTINEL in proc.stderr


# --------------------------------------------------------------------------
# ordering guards -- these encode kernel requirements, not style preferences
# --------------------------------------------------------------------------


@requires_linux_x86
def test_seccomp_requires_no_new_privs_first():
    """Without CAP_SYS_ADMIN the kernel refuses a filter unless no_new_privs is
    already set. If this ever stops being true the ordering in apply_lockdown
    is no longer load-bearing and the comment there should be revisited."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes, sys\n"
            "sys.path.insert(0, %r)\n" % str(Path(sandbox.__file__).parent.parent)
            + "from syft_job.sandbox import _build_filter, _libc, "
            "_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER\n"
            "libc = _libc()\n"
            "prog, _keep = _build_filter()\n"
            "rc = libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, "
            "ctypes.cast(ctypes.byref(prog), ctypes.c_void_p), 0, 0)\n"
            "print('rc', rc)\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Non-root without CAP_SYS_ADMIN: must fail. If running as root with the
    # capability the call may succeed, so only assert the failure when we are
    # genuinely unprivileged.
    if os.getuid() != 0:
        assert proc.stdout.strip() == "rc -1", proc.stdout + proc.stderr


def test_setgid_must_precede_setuid_is_documented():
    """Guards against a refactor silently reordering the privilege drop."""
    src = Path(sandbox.__file__).read_text()
    drop = src[src.index("os.setgroups([])") : src.index("if os.getuid() != uid")]
    assert drop.index("os.setgroups") < drop.index("os.setgid") < drop.index(
        "os.setuid"
    ), "privilege drop order changed: setgroups -> setgid -> setuid is required"


# --------------------------------------------------------------------------
# integration: the lockdown must actually engage through the real runner
# --------------------------------------------------------------------------

NETWORK_PROBE_MAIN_PY = """\
import os, socket

os.makedirs("outputs", exist_ok=True)
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    verdict = "NETWORK_ALLOWED"
except OSError:
    verdict = "NETWORK_BLOCKED"
with open("outputs/verdict.txt", "w") as f:
    f.write(verdict)
print(verdict)
"""


def _run_probe_job(tmp_path: Path) -> str:
    """Submit and run a job that reports whether it can make a socket."""
    from syft_job.client import JobClient
    from syft_job.config import SyftJobConfig
    from syft_job.job_runner import SyftJobRunner

    do_email, ds_email = "do@test.org", "ds@test.org"
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(NETWORK_PROBE_MAIN_PY)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=do_email)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=ds_email)
    JobClient(config=ds_config).submit_python_job(
        user=do_email, code_path=str(code_file), job_name="probe.job"
    )
    do_client = JobClient(config=do_config)
    do_client.jobs[0].approve()
    SyftJobRunner(config=do_config).process_approved_jobs(
        stream_output=False, timeout=900
    )

    review = do_config.get_review_job_dir(do_email, ds_email, "probe.job")
    verdict_file = review / "outputs" / "verdict.txt"
    if not verdict_file.exists():
        stderr = (review / "stderr.txt").read_text() if (
            review / "stderr.txt"
        ).exists() else "<no stderr>"
        raise AssertionError(f"job produced no verdict; stderr:\n{stderr[-3000:]}")
    return verdict_file.read_text().strip()


@requires_linux_x86
@pytest.mark.slow
def test_job_has_network_without_sandbox(tmp_path, monkeypatch):
    """Control: confirms the probe would otherwise succeed."""
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "off")
    assert _run_probe_job(tmp_path) == "NETWORK_ALLOWED"


@requires_linux_x86
@pytest.mark.slow
def test_job_network_blocked_with_sandbox(tmp_path, monkeypatch):
    """The whole point: a real job, run through the real runner, cannot network."""
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "require")
    monkeypatch.setenv(job_runner.SANDBOX_UID_ENV_VAR, str(os.getuid()))
    monkeypatch.setenv(job_runner.SANDBOX_GID_ENV_VAR, str(os.getgid()))
    assert _run_probe_job(tmp_path) == "NETWORK_BLOCKED"


# --------------------------------------------------------------------------
# partial lockdown must not pass silently
# --------------------------------------------------------------------------


@requires_linux_x86
@pytest.mark.skipif(os.getuid() == 0, reason="needs to run as a non-root user")
def test_refuses_when_privileges_cannot_be_dropped():
    """As a non-root user the uid drop is impossible.

    The seccomp half would still apply, but the job would keep this user's file
    access -- so `require` must refuse rather than half-protect.
    """
    proc = subprocess.run(
        [sys.executable, SANDBOX_PY, "--uid", "65534", "--", "echo", "SHOULD_NOT_RUN"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == sandbox.REFUSED_EXIT_CODE
    assert "cannot drop to uid 65534" in proc.stderr
    assert "SHOULD_NOT_RUN" not in proc.stdout


@requires_linux_x86
@pytest.mark.skipif(os.getuid() == 0, reason="needs to run as a non-root user")
def test_best_effort_proceeds_but_warns():
    proc = subprocess.run(
        [
            sys.executable,
            SANDBOX_PY,
            "--uid",
            "65534",
            "--best-effort",
            "--",
            sys.executable,
            "-c",
            "import socket\n"
            "try:\n"
            "    socket.socket(2, 1); print('ALLOWED')\n"
            "except OSError: print('BLOCKED')\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "BLOCKED"  # filter still applied
    assert "WARNING" in proc.stderr  # but the partial application is announced


@requires_linux_x86
def test_require_mode_does_not_pass_best_effort(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "require")
    assert "--best-effort" not in job_runner.build_job_command(Path("/tmp/run.sh"))


@requires_linux_x86
def test_on_mode_passes_best_effort(monkeypatch):
    monkeypatch.setenv(job_runner.SANDBOX_ENV_VAR, "on")
    assert "--best-effort" in job_runner.build_job_command(Path("/tmp/run.sh"))
