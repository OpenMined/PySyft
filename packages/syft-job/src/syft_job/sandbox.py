"""Kernel-enforced lockdown for untrusted job code.

Run as ``python -m syft_job.sandbox -- <command> [args...]``. The module drops
privileges, installs a seccomp filter that denies socket creation, and then
``execve``s the command. Both restrictions are one-way: the kernel offers no
operation to remove a seccomp filter or to undo ``PR_SET_NO_NEW_PRIVS``, so the
code being launched cannot lift them.

Because the filter is enforced by the kernel on syscall entry and survives
``execve``, it binds compiled binaries -- C/C++/CUDA -- exactly as it binds
Python. Blocking socket *creation* rather than ``connect`` is deliberate: a
seccomp filter cannot dereference pointers, so it cannot inspect the address
passed to ``connect``, but ``socket``'s arguments are plain integers. Denying
every address family (not just AF_INET) also prevents the process from
receiving an already-connected socket over a unix socket via SCM_RIGHTS.

``execvp`` is the last statement, so any failure above it means the command is
never executed -- the sandbox fails closed by construction.

Linux/x86-64 only. ``is_supported()`` reports availability so callers can
degrade or refuse deliberately rather than silently running unprotected.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import sys

__all__ = [
    "is_supported",
    "apply_lockdown",
    "main",
    "SandboxError",
    "REFUSED_EXIT_CODE",
    "SENTINEL",
]

# Exit code used when the sandbox refuses to run the command. Distinct from any
# plausible exit code of the job itself so the caller can tell "we refused" from
# "the job failed".
REFUSED_EXIT_CODE = 93

# Printed to stderr on refusal; callers grep for this.
SENTINEL = "SYFT_SANDBOX_REFUSED"

# Default unprivileged account. Overridable for tests and for images that ship a
# dedicated job user.
DEFAULT_UID = 65534
DEFAULT_GID = 65534

# prctl(2) options.
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

# Classic BPF opcodes used by the filter.
_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_K = 0x00
_BPF_RET = 0x06

_AUDIT_ARCH_X86_64 = 0xC000003E

_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000

# Offsets into struct seccomp_data.
_OFF_NR = 0
_OFF_ARCH = 4

# x86-64 syscall numbers denied by the filter.
#
# socket/socketpair are the only ways to obtain a socket descriptor, so denying
# them denies networking outright. io_uring can perform network I/O through its
# submission queue without issuing the corresponding syscalls, which a classic
# seccomp filter cannot observe, so it must be denied too. The ptrace family is
# denied to stop the job inspecting or modifying other processes.
_DENIED_SYSCALLS = (
    41,  # socket
    53,  # socketpair
    425,  # io_uring_setup
    426,  # io_uring_enter
    101,  # ptrace
    310,  # process_vm_readv
    311,  # process_vm_writev
)


class SandboxError(RuntimeError):
    """The lockdown could not be applied."""


class _SockFilter(ctypes.Structure):
    """struct sock_filter -- one classic-BPF instruction."""

    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint),
    ]


class _SockFprog(ctypes.Structure):
    """struct sock_fprog -- the program handed to the kernel."""

    _fields_ = [
        ("len", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    # Without explicit argtypes ctypes passes the struct pointer as a 32-bit int
    # on x86-64 and the kernel reads a truncated address.
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    return libc


def _build_filter() -> tuple[_SockFprog, ctypes.Array]:
    """Assemble the BPF program.

    Returns the program and the instruction array. The caller must keep the
    array alive until prctl returns: ``_SockFprog`` holds a raw pointer that the
    garbage collector does not treat as a reference.
    """
    prog: list[_SockFilter] = []

    prog.append(_SockFilter(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, _OFF_ARCH))
    arch_at = len(prog)
    prog.append(_SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, 0, 0, _AUDIT_ARCH_X86_64))

    prog.append(_SockFilter(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, _OFF_NR))
    tests_at: list[int] = []
    for nr in _DENIED_SYSCALLS:
        tests_at.append(len(prog))
        prog.append(_SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, 0, 0, nr))

    allow_at = len(prog)
    prog.append(_SockFilter(_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_ALLOW))
    deny_at = len(prog)
    prog.append(
        _SockFilter(_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM)
    )

    # Jump targets are offsets relative to the instruction *after* the jump, not
    # absolute indices. Getting this wrong routes ordinary syscalls into the
    # deny branch, which crashes the process immediately.
    prog[arch_at].jt = 0
    prog[arch_at].jf = deny_at - arch_at - 1
    for i in tests_at:
        prog[i].jt = deny_at - i - 1
        prog[i].jf = 0

    if allow_at != deny_at - 1:
        raise SandboxError("filter assembly produced an unexpected layout")

    instructions = (_SockFilter * len(prog))(*prog)
    return _SockFprog(len(prog), instructions), instructions


def is_supported() -> tuple[bool, str]:
    """Whether the lockdown can be applied here.

    Returns ``(supported, reason)``; ``reason`` is empty when supported.
    """
    if sys.platform != "linux":
        return False, f"requires Linux, running on {sys.platform}"
    if platform.machine() not in ("x86_64", "AMD64"):
        return False, f"filter is x86-64 only, running on {platform.machine()}"
    try:
        _libc()
    except OSError as exc:
        return False, f"cannot load libc: {exc}"
    return True, ""


def apply_lockdown(uid: int = DEFAULT_UID, gid: int = DEFAULT_GID) -> None:
    """Drop privileges and install the seccomp filter, in that order.

    Raises ``SandboxError`` if any step fails. On return the calling process
    cannot create sockets, cannot regain privilege, and cannot undo either.

    The ordering is enforced by the kernel, not chosen for style:

    * ``setgid`` must precede ``setuid`` -- after dropping the user id the
      process is no longer permitted to change its group, so a reversed order
      leaves the job in the root group.
    * ``PR_SET_NO_NEW_PRIVS`` must precede ``PR_SET_SECCOMP`` -- installing a
      filter otherwise requires CAP_SYS_ADMIN, which the enclave container does
      not have.
    """
    supported, reason = is_supported()
    if not supported:
        raise SandboxError(reason)

    libc = _libc()

    if os.getuid() == 0:
        try:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
        except OSError as exc:
            raise SandboxError(f"privilege drop failed: {exc}") from exc
        if os.getuid() != uid or os.geteuid() != uid:
            raise SandboxError("privilege drop did not take effect")
    elif os.getuid() != uid:
        # Already unprivileged. Continue -- the filter is still worth applying --
        # but do not pretend we dropped to the requested account.
        pass

    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, None, 0, 0) != 0:
        raise SandboxError(
            f"PR_SET_NO_NEW_PRIVS failed: {os.strerror(ctypes.get_errno())}"
        )

    prog, _instructions = _build_filter()
    rc = libc.prctl(
        _PR_SET_SECCOMP,
        _SECCOMP_MODE_FILTER,
        ctypes.cast(ctypes.byref(prog), ctypes.c_void_p),
        0,
        0,
    )
    if rc != 0:
        raise SandboxError(
            f"PR_SET_SECCOMP failed: {os.strerror(ctypes.get_errno())}"
        )


def _refuse(reason: str) -> "None":
    print(f"{SENTINEL}: {reason}", file=sys.stderr, flush=True)
    # os._exit avoids running atexit handlers or flushing inherited buffers in
    # a process that is midway through dropping privileges.
    os._exit(REFUSED_EXIT_CODE)


def main(argv: list[str] | None = None) -> None:
    """Entry point: apply the lockdown, then become the requested command."""
    args = list(sys.argv[1:] if argv is None else argv)

    uid, gid = DEFAULT_UID, DEFAULT_GID
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--":
            break
        if flag in ("--uid", "--gid"):
            if not args:
                _refuse(f"{flag} requires a value")
            try:
                value = int(args.pop(0))
            except ValueError:
                _refuse(f"{flag} requires an integer")
            if flag == "--uid":
                uid = value
            else:
                gid = value
        else:
            _refuse(f"unknown option {flag}")

    if not args:
        _refuse("no command given")

    try:
        apply_lockdown(uid=uid, gid=gid)
    except SandboxError as exc:
        _refuse(str(exc))

    try:
        os.execvp(args[0], args)
    except OSError as exc:
        _refuse(f"exec {args[0]!r} failed: {exc}")


if __name__ == "__main__":
    main()
