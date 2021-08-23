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

# !mitogen: minify_safe

"""
mitogen.profiler
    Record and report cProfile statistics from a run. Creates one aggregated
    output file, one aggregate containing only workers, and one for the
    top-level process.

Usage:
    mitogen.profiler record <dest_path> <tool> [args ..]
    mitogen.profiler report <dest_path> [sort_mode]
    mitogen.profiler stat <sort_mode> <tool> [args ..]

Mode:
    record: Record a trace.
    report: Report on a previously recorded trace.
    stat: Record and report in a single step.

Where:
    dest_path: Filesystem prefix to write .pstats files to.
    sort_mode: Sorting mode; defaults to "cumulative". See:
        https://docs.python.org/2/library/profile.html#pstats.Stats.sort_stats

Example:
    mitogen.profiler record /tmp/mypatch ansible-playbook foo.yml
    mitogen.profiler dump /tmp/mypatch-worker.pstats
"""

# future
from __future__ import print_function

# stdlib
import os
import pstats
import shutil
import subprocess
import sys
import tempfile
import time


def try_merge(stats, path):
    try:
        stats.add(path)
        return True
    except Exception as e:
        print("%s failed. Will retry. %s" % (path, e))
        return False


def merge_stats(outpath, inpaths):
    first, rest = inpaths[0], inpaths[1:]
    for x in range(1):
        try:
            stats = pstats.Stats(first)
        except EOFError:
            time.sleep(0.2)
            continue

        print("Writing %r..." % (outpath,))
        for path in rest:
            # print("Merging %r into %r.." % (os.path.basename(path), outpath))
            for x in range(5):
                if try_merge(stats, path):
                    break
                time.sleep(0.2)

    stats.dump_stats(outpath)


def generate_stats(outpath, tmpdir):
    print("Generating stats..")
    all_paths = []
    paths_by_ident = {}

    for name in os.listdir(tmpdir):
        if name.endswith("-dump.pstats"):
            ident, _, pid = name.partition("-")
            path = os.path.join(tmpdir, name)
            all_paths.append(path)
            paths_by_ident.setdefault(ident, []).append(path)

    merge_stats("%s-all.pstat" % (outpath,), all_paths)
    for ident, paths in paths_by_ident.items():
        merge_stats("%s-%s.pstat" % (outpath, ident), paths)


def do_record(tmpdir, path, *args):
    env = os.environ.copy()
    fmt = "%(identity)s-%(pid)s.%(now)s-dump.%(ext)s"
    env["MITOGEN_PROFILING"] = "1"
    env["MITOGEN_PROFILE_FMT"] = os.path.join(tmpdir, fmt)
    rc = subprocess.call(args, env=env)
    generate_stats(path, tmpdir)
    return rc


def do_report(tmpdir, path, sort="cumulative"):
    stats = pstats.Stats(path).sort_stats(sort)
    stats.print_stats(100)


def do_stat(tmpdir, sort, *args):
    valid_sorts = pstats.Stats.sort_arg_dict_default
    if sort not in valid_sorts:
        sys.stderr.write(
            "Invalid sort %r, must be one of %s\n"
            % (sort, ", ".join(sorted(valid_sorts)))
        )
        sys.exit(1)

    outfile = os.path.join(tmpdir, "combined")
    do_record(tmpdir, outfile, *args)
    aggs = (
        "app.main",
        "mitogen.broker",
        "mitogen.child_main",
        "mitogen.service.pool",
        "Strategy",
        "WorkerProcess",
        "all",
    )
    for agg in aggs:
        path = "%s-%s.pstat" % (outfile, agg)
        if os.path.exists(path):
            print()
            print()
            print("------ Aggregation %r ------" % (agg,))
            print()
            do_report(tmpdir, path, sort)
            print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "report", "stat"):
        sys.stderr.write(__doc__.lstrip())
        sys.exit(1)

    func = globals()["do_" + sys.argv[1]]
    tmpdir = tempfile.mkdtemp(prefix="mitogen.profiler")
    try:
        sys.exit(func(tmpdir, *sys.argv[2:]) or 0)
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
