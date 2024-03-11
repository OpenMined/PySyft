"""
Capture stdout, stderr and stdin streams

References:
- https://github.com/OpenMined/PySyft/pull/8560
- https://github.com/pytest-dev/py/blob/master/py/_io/capture.py
"""

# stdlib
from collections.abc import Callable
from collections.abc import Generator
import contextlib
import os
import sys
import tempfile
from typing import Any
from typing import cast

patchsysdict = {0: "stdin", 1: "stdout", 2: "stderr"}

try:
    devnullpath = os.devnull
except AttributeError:
    if os.name == "nt":
        devnullpath = "NUL"
    else:
        devnullpath = "/dev/null"


class DontReadFromInput:
    """Temporary stub class.  Ideally when stdin is accessed, the
    capturing should be turned off, with possibly all data captured
    so far sent to the screen.  This should be configurable, though,
    because in automated test runs it is better to crash than
    hang indefinitely.
    """

    def read(self, *args: Any) -> None:
        raise OSError("reading from stdin while output is captured")

    readline = read
    readlines = read
    __iter__ = read

    def fileno(self) -> None:
        raise ValueError("redirected Stdin is pseudofile, has no fileno()")

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass


class Capture:
    @classmethod
    def call(cls, func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, str, str]:
        """return a (res, out, err) tuple where
        out and err represent the output/error output
        during function execution.
        call the given function with args/kwargs
        and capture output/error during its execution.
        """
        so = cls()
        try:
            res = func(*args, **kwargs)
        finally:
            out, err = so.reset()
        return res, out, err

    def reset(self) -> tuple[str, str]:
        """reset sys.stdout/stderr and return captured output as strings."""
        if hasattr(self, "_reset"):
            raise ValueError("was already reset")
        self._reset = True
        outfile, errfile = self.done(save=False)
        out, err = "", ""
        if outfile and not outfile.closed:
            out = outfile.read()
            outfile.close()
        if errfile and errfile != outfile and not errfile.closed:
            err = errfile.read()
            errfile.close()
        return out, err

    def suspend(self) -> tuple[str, str]:
        """return current snapshot captures, memorize tempfiles."""
        outerr = self.readouterr()
        outfile, errfile = self.done()
        return outerr


class FDCapture:
    """Capture IO to/from a given os-level filedescriptor."""

    def __init__(
        self,
        targetfd: int,
        tmpfile: Any | None = None,
        now: bool = True,
        patchsys: bool = False,
    ) -> None:
        """save targetfd descriptor, and open a new
        temporary file there.  If no tmpfile is
        specified a tempfile.Tempfile() will be opened
        in text mode.
        """
        self.targetfd = targetfd
        if tmpfile is None and targetfd != 0:
            f = tempfile.TemporaryFile("wb+")
            tmpfile = dupfile(f, encoding="UTF-8")
            f.close()
        self.tmpfile = cast(Any, tmpfile)
        self._savefd = os.dup(self.targetfd)
        if patchsys:
            self._oldsys = getattr(sys, patchsysdict[targetfd])
        if now:
            self.start()

    def start(self) -> None:
        try:
            os.fstat(self._savefd)
        except OSError:
            raise ValueError(
                "saved filedescriptor not valid, " "did you call start() twice?"
            )
        if self.targetfd == 0 and not self.tmpfile:
            fd = os.open(devnullpath, os.O_RDONLY)
            os.dup2(fd, 0)
            os.close(fd)
            if hasattr(self, "_oldsys"):
                setattr(sys, patchsysdict[self.targetfd], DontReadFromInput())
        else:
            os.dup2(self.tmpfile.fileno(), self.targetfd)
            if hasattr(self, "_oldsys"):
                setattr(sys, patchsysdict[self.targetfd], self.tmpfile)

    def done(self) -> Any:
        """unpatch and clean up, returns the self.tmpfile (file object)"""
        os.dup2(self._savefd, self.targetfd)
        os.close(self._savefd)
        if self.targetfd != 0:
            self.tmpfile.seek(0)
        if hasattr(self, "_oldsys"):
            setattr(sys, patchsysdict[self.targetfd], self._oldsys)
        return self.tmpfile

    def writeorg(self, data: bytes) -> None:
        """write a string to the original file descriptor"""
        tempfp = tempfile.TemporaryFile()
        try:
            os.dup2(self._savefd, tempfp.fileno())
            tempfp.write(data)
        finally:
            tempfp.close()


class StdCaptureFD(Capture):
    """This class allows to capture writes to FD1 and FD2
    and may connect a NULL file to FD0 (and prevent
    reads from sys.stdin).  If any of the 0,1,2 file descriptors
    is invalid it will not be captured.
    """

    def __init__(
        self,
        out: bool = True,
        err: bool = True,
        mixed: bool = False,
        in_: bool = True,
        patchsys: bool = True,
        now: bool = True,
    ):
        self._options = {
            "out": out,
            "err": err,
            "mixed": mixed,
            "in_": in_,
            "patchsys": patchsys,
            "now": now,
        }
        self._save()
        if now:
            self.startall()

    def _save(self) -> None:
        in_ = self._options["in_"]
        out = self._options["out"]
        err = self._options["err"]
        mixed = self._options["mixed"]
        patchsys = self._options["patchsys"]
        if in_:
            try:
                self.in_ = FDCapture(0, tmpfile=None, now=False, patchsys=patchsys)
            except OSError:
                pass
        if out:
            tmpfile = None
            if hasattr(out, "write"):
                tmpfile = out
            try:
                self.out = FDCapture(1, tmpfile=tmpfile, now=False, patchsys=patchsys)
                self._options["out"] = self.out.tmpfile
            except OSError:
                pass
        if err:
            if out and mixed:
                tmpfile = self.out.tmpfile
            elif hasattr(err, "write"):
                tmpfile = err
            else:
                tmpfile = None
            try:
                self.err = FDCapture(2, tmpfile=tmpfile, now=False, patchsys=patchsys)
                self._options["err"] = self.err.tmpfile
            except OSError:
                pass

    def startall(self) -> None:
        if hasattr(self, "in_"):
            self.in_.start()
        if hasattr(self, "out"):
            self.out.start()
        if hasattr(self, "err"):
            self.err.start()

    def resume(self) -> None:
        """resume capturing with original temp files."""
        self.startall()

    def done(self, save: bool = True) -> tuple[Any | None, Any | None]:
        """return (outfile, errfile) and stop capturing."""
        outfile = errfile = None
        if hasattr(self, "out") and not self.out.tmpfile.closed:
            outfile = self.out.done()
        if hasattr(self, "err") and not self.err.tmpfile.closed:
            errfile = self.err.done()
        if hasattr(self, "in_"):
            self.in_.done()
        if save:
            self._save()
        return outfile, errfile

    def readouterr(self) -> tuple[str, str]:
        """return snapshot value of stdout/stderr capturings."""
        if hasattr(self, "out"):
            out = self._readsnapshot(self.out.tmpfile)
        else:
            out = ""
        if hasattr(self, "err"):
            err = self._readsnapshot(self.err.tmpfile)
        else:
            err = ""
        return out, err

    def _readsnapshot(self, f: Any) -> str:
        f.seek(0)
        res = f.read()
        enc = getattr(f, "encoding", None)
        if enc:

            def _totext(
                obj: Any, encoding: str | None = None, errors: str | None = None
            ) -> str:
                """
                Source: https://github.com/pytest-dev/py/blob/master/py/_builtin.py
                """
                if isinstance(obj, bytes):
                    if errors is None:
                        obj = obj.decode(encoding)
                    else:
                        obj = obj.decode(encoding, errors)
                elif not isinstance(obj, str):
                    obj = str(obj)
                return obj

            res = _totext(res, enc, "replace")
        f.truncate(0)
        f.seek(0)
        return res


def dupfile(
    f: Any,
    mode: str | None = None,
    buffering: int = 0,
    raising: bool = False,
    encoding: str | None = None,
) -> Any:
    """return a new open file object that's a duplicate of f

    mode is duplicated if not given, 'buffering' controls
    buffer size (defaulting to no buffering) and 'raising'
    defines whether an exception is raised when an incompatible
    file object is passed in (if raising is False, the file
    object itself will be returned)
    """
    try:
        fd = f.fileno()
        mode = mode or f.mode
        newfd = os.dup(fd)
    except AttributeError:
        if raising:
            raise
        return f

    if encoding is not None:
        mode = mode.replace("b", "")
        buffering = True

    return os.fdopen(newfd, mode, buffering, encoding, closefd=True)


@contextlib.contextmanager
def std_stream_capture(out: bool = True, err: bool = True) -> Generator[Any, None, Any]:
    try:
        capture = StdCaptureFD(out=out, err=err)
    except Exception:
        capture = None

    try:
        yield
    finally:
        if capture is not None:
            capture.reset()
