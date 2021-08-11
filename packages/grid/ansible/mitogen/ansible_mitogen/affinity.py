# Copyright 2019, David Wilson
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
As Mitogen separates asynchronous IO out to a broker thread, communication
necessarily involves context switching and waking that thread. When application
threads and the broker share a CPU, this can be almost invisibly fast - around
25 microseconds for a full A->B->A round-trip.

However when threads are scheduled on different CPUs, round-trip delays
regularly vary wildly, and easily into milliseconds. Many contributing factors
exist, not least scenarios like:

1. A is preempted immediately after waking B, but before releasing the GIL.
2. B wakes from IO wait only to immediately enter futex wait.
3. A may wait 10ms or more for another timeslice, as the scheduler on its CPU
   runs threads unrelated to its transaction (i.e. not B), wake only to release
   its GIL, before entering IO sleep waiting for a reply from B, which cannot
   exist yet.
4. B wakes, acquires GIL, performs work, and sends reply to A, causing it to
   wake. B is preempted before releasing GIL.
5. A wakes from IO wait only to immediately enter futex wait.
6. B may wait 10ms or more for another timeslice, wake only to release its GIL,
   before sleeping again.
7. A wakes, acquires GIL, finally receives reply.

Per above if we are unlucky, on an even moderately busy machine it is possible
to lose milliseconds just in scheduling delay, and the effect is compounded
when pairs of threads in process A are communicating with pairs of threads in
process B using the same scheme, such as when Ansible WorkerProcess is
communicating with ContextService in the connection multiplexer. In the worst
case it could involve 4 threads working in lockstep spread across 4 busy CPUs.

Since multithreading in Python is essentially useless except for waiting on IO
due to the presence of the GIL, at least in Ansible there is no good reason for
threads in the same process to run on distinct CPUs - they always operate in
lockstep due to the GIL, and are thus vulnerable to issues like above.

Linux lacks any natural API to describe what we want, it only permits
individual threads to be constrained to run on specific CPUs, and for that
constraint to be inherited by new threads and forks of the constrained thread.

This module therefore implements a CPU pinning policy for Ansible processes,
providing methods that should be called early in any new process, either to
rebalance which CPU it is pinned to, or in the case of subprocesses, to remove
the pinning entirely. It is likely to require ongoing tweaking, since pinning
necessarily involves preventing the scheduler from making load balancing
decisions.
"""

# future
from __future__ import absolute_import

# stdlib
import ctypes
import logging
import mmap
import multiprocessing
import os
import struct

# third party
import mitogen.core
import mitogen.parent

LOG = logging.getLogger(__name__)


try:
    _libc = ctypes.CDLL(None, use_errno=True)
    _strerror = _libc.strerror
    _strerror.restype = ctypes.c_char_p
    _sem_init = _libc.sem_init
    _sem_wait = _libc.sem_wait
    _sem_post = _libc.sem_post
    _sched_setaffinity = _libc.sched_setaffinity
except (OSError, AttributeError):
    _libc = None
    _strerror = None
    _sem_init = None
    _sem_wait = None
    _sem_post = None
    _sched_setaffinity = None


class sem_t(ctypes.Structure):
    """
    Wrap sem_t to allow storing a lock in shared memory.
    """

    _fields_ = [
        ("data", ctypes.c_uint8 * 128),
    ]

    def init(self):
        if _sem_init(self.data, 1, 1):
            raise Exception(_strerror(ctypes.get_errno()))

    def acquire(self):
        if _sem_wait(self.data):
            raise Exception(_strerror(ctypes.get_errno()))

    def release(self):
        if _sem_post(self.data):
            raise Exception(_strerror(ctypes.get_errno()))


class State(ctypes.Structure):
    """
    Contents of shared memory segment. This allows :meth:`Manager.assign` to be
    called from any child, since affinity assignment must happen from within
    the context of the new child process.
    """

    _fields_ = [
        ("lock", sem_t),
        ("counter", ctypes.c_uint8),
    ]


class Policy(object):
    """
    Process affinity policy.
    """

    def assign_controller(self):
        """
        Assign the Ansible top-level policy to this process.
        """

    def assign_muxprocess(self, index):
        """
        Assign the MuxProcess policy to this process.
        """

    def assign_worker(self):
        """
        Assign the WorkerProcess policy to this process.
        """

    def assign_subprocess(self):
        """
        Assign the helper subprocess policy to this process.
        """


class FixedPolicy(Policy):
    """
    :class:`Policy` for machines where the only control method available is
    fixed CPU placement. The scheme here was tested on an otherwise idle 16
    thread machine.

    - The connection multiplexer is pinned to CPU 0.
    - The Ansible top-level (strategy) is pinned to CPU 1.
    - WorkerProcesses are pinned sequentually to 2..N, wrapping around when no
      more CPUs exist.
    - Children such as SSH may be scheduled on any CPU except 0/1.

    If the machine has less than 4 cores available, the top-level and workers
    are pinned between CPU 2..N, i.e. no CPU is reserved for the top-level
    process.

    This could at least be improved by having workers pinned to independent
    cores, before reusing the second hyperthread of an existing core.

    A hook is installed that causes :meth:`reset` to run in the child of any
    process created with :func:`mitogen.parent.popen`, ensuring CPU-intensive
    children like SSH are not forced to share the same core as the (otherwise
    potentially very busy) parent.
    """

    def __init__(self, cpu_count=None):
        #: For tests.
        self.cpu_count = cpu_count or multiprocessing.cpu_count()
        self.mem = mmap.mmap(-1, 4096)
        self.state = State.from_buffer(self.mem)
        self.state.lock.init()

        if self.cpu_count < 2:
            # uniprocessor
            self._reserve_mux = False
            self._reserve_controller = False
            self._reserve_mask = 0
            self._reserve_shift = 0
        elif self.cpu_count < 4:
            # small SMP
            self._reserve_mux = True
            self._reserve_controller = False
            self._reserve_mask = 1
            self._reserve_shift = 1
        else:
            # big SMP
            self._reserve_mux = True
            self._reserve_controller = True
            self._reserve_mask = 3
            self._reserve_shift = 2

    def _set_affinity(self, descr, mask):
        if descr:
            LOG.debug("CPU mask for %s: %#08x", descr, mask)
        mitogen.parent._preexec_hook = self._clear
        self._set_cpu_mask(mask)

    def _balance(self, descr):
        self.state.lock.acquire()
        try:
            n = self.state.counter
            self.state.counter += 1
        finally:
            self.state.lock.release()

        self._set_cpu(
            descr, self._reserve_shift + ((n % (self.cpu_count - self._reserve_shift)))
        )

    def _set_cpu(self, descr, cpu):
        self._set_affinity(descr, 1 << (cpu % self.cpu_count))

    def _clear(self):
        all_cpus = (1 << self.cpu_count) - 1
        self._set_affinity(None, all_cpus & ~self._reserve_mask)

    def assign_controller(self):
        if self._reserve_controller:
            self._set_cpu("Ansible top-level process", 1)
        else:
            self._balance("Ansible top-level process")

    def assign_muxprocess(self, index):
        self._set_cpu("MuxProcess %d" % (index,), index)

    def assign_worker(self):
        self._balance("WorkerProcess")

    def assign_subprocess(self):
        self._clear()


class LinuxPolicy(FixedPolicy):
    def _mask_to_bytes(self, mask):
        """
        Convert the (type long) mask to a cpu_set_t.
        """
        chunks = []
        shiftmask = (2 ** 64) - 1
        for x in range(16):
            chunks.append(struct.pack("<Q", mask & shiftmask))
            mask >>= 64
        return mitogen.core.b("").join(chunks)

    def _get_thread_ids(self):
        try:
            ents = os.listdir("/proc/self/task")
        except OSError:
            LOG.debug("cannot fetch thread IDs for current process")
            return [os.getpid()]

        return [int(s) for s in ents if s.isdigit()]

    def _set_cpu_mask(self, mask):
        s = self._mask_to_bytes(mask)
        for tid in self._get_thread_ids():
            _sched_setaffinity(tid, len(s), s)


if _sched_setaffinity is not None:
    policy = LinuxPolicy()
else:
    policy = Policy()
