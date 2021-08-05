import collections
import threading
import warnings

class Pos(collections.namedtuple('Pos', 'fname line col log_fp')):
    """A position in a file.

    This also optionally tracks a file-like object for logging
    warnings and errors associated with this file.
    """

    def __str__(self):
        return '{}:{}:{}'.format(self.fname, self.line, self.col)

    def warn(self, msg):
        """Log msg to this Pos's logger.

        If log_fp is None, the warning is silently discarded.
        """
        if self.log_fp is not None:
            self.log_fp.write('{}: warning: {}\n'.format(self, msg))

    def raise_error(self, msg):
        """Log and raise InputError([(self, msg)])."""
        if self.log_fp is not None:
            self.log_fp.write('{}: error: {}\n'.format(self, msg))
        raise InputError([(self, msg)])

Pos.unknown = Pos('<unknown>', 1, 0, None)

class PosFactory:
    """A factory that translates character offsets to Pos instances."""

    def __init__(self, fname, string, log_fp=None):
        self.__fname = fname
        self.__string = string
        self.__log_fp = log_fp
        self.__cache = (0, 1, 0)

    def offset_to_pos(self, offset):
        last_off, last_line, last_col = self.__cache
        if last_off < offset:
            last_off, last_line, last_col = 0, 1, 0

        line = self.__string.count('\n', last_off, offset) + last_line
        lastnl = self.__string.rfind('\n', last_off, offset)
        if lastnl == -1:
            col = last_col + (offset - last_off)
        else:
            col = offset - lastnl - 1
        self.__cache = (offset, line, col)

        return Pos(self.__fname, line, col, self.__log_fp)

class InputError(ValueError):
    """One or more errors with associated Pos instances.

    An InputError representing multiple errors is a "bundled" error.
    These can be created when recovering from errors with an
    InputErrorRecoverer.
    """

    def __init__(self, list_of_pos_msg):
        super().__init__(list_of_pos_msg)

    def __str__(self):
        list_of_msg_pos = self.args[0]
        if len(list_of_msg_pos) == 1:
            return '1 error'
        return '{} errors'.format(len(list_of_msg_pos))

class InputErrorRecoverer:
    """A context manager for recovering from and bundling InputErrors.

    This context manager catches and collects InputErrors, effectively
    recovering from InputErrors at the end of the with block.  An
    InputErrorRecoverer can be used several times, after which
    collected errors can be re-raised as a bundled InputError.  For
    example, a caller that wishes to parse several files without
    stopping because of an error in one file can do something like
    this:

        recoverer = InputErrorRecoverer()
        for filename in filenames:
            with recoverer:
                parse(filename)
        recoverer.reraise()

    An InputErrorRecoverer *must* be either reraised or disposed (even
    if no errors occurred).  Otherwise a UserWarning will be issued.
    """

    def __init__(self):
        self.__errors = []

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__errors is None:
            raise ValueError('InputErrorRecoverer already disposed')
        if isinstance(exc_value, InputError):
            self.__errors.extend(exc_value.args)
            return True

    def __del__(self):
        if self.__errors is not None:
            try:
                warnings.warn('InputErrorRecoverer must be reraised or disposed',
                              stacklevel=2)
            except TypeError as e:
                # If Python is exiting, warnings.warn has a habit of
                # raising TypeError("'NoneType' object is not
                # iterable",).  Ignore it.
                pass

    def reraise(self):
        """If any errors have been collected, raise a bundled InputError."""
        errors = self.__errors
        self.dispose()
        if errors:
            raise InputError(errors)

    def dispose(self):
        """Discard all collected errors."""
        self.__errors = None
