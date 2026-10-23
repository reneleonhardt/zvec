#!/usr/bin/env python3
"""Attribute cold-index syscalls to preparation, open, search and cleanup.

The existing executable reads /proc/self/io at the three phase boundaries.
Tracing is diagnostic only; paired timing runs never run under strace.
"""

import bisect
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def events(lines):
    pending = {}
    for line in lines:
        match = re.match(r"(\d+)\s+(\d+\.\d+)\s+(.*)", line)
        if not match:
            continue
        pid, timestamp, call = match.groups()
        if "<unfinished ...>" in call:
            pending[pid] = (float(timestamp), call.split("<unfinished ...>")[0])
            continue
        if call.startswith("<... "):
            if pid not in pending:
                raise ValueError("resumed syscall without its entry")
            timestamp, prefix = pending.pop(pid)
            call = prefix + call.split(" resumed>", 1)[1]
        match = re.match(r"(\w+)\((.*)\)\s+=\s+(.*?)\s+<(\d+\.\d+)>$", call)
        if match:
            name, args, result, duration = match.groups()
            yield float(timestamp), name, args, result, float(duration)
    if pending:
        raise ValueError("unfinished syscall at end of trace")


def summarize(trace):
    calls = sorted(events(trace), key=lambda event: event[0])
    boundaries = [time for time, name, args, _, _ in calls
                  if name == "openat" and '"/proc/self/io"' in args]
    if not boundaries or len(boundaries) % 3:
        raise ValueError(f"expected three I/O markers per trial, found {len(boundaries)}")
    trials = [{} for _ in range(len(boundaries) // 3)]
    preparations = [(time, int(match[1]))
                    for time, name, args, _, _ in calls
                    if name in ("link", "linkat")
                    and (match := re.search(r"/cold-\d+-(\d+)\.index", args))]
    descriptors = {}
    for time, name, args, result, duration in calls:
        fd = re.match(r"(-?\d+)(?:<[^>]*>)?(?:,|$)", args)
        channel = descriptors.get(int(fd[1]), "other") if fd else "other"
        if name == "openat" and "/cold-" in args:
            channel = "direct" if "O_DIRECT" in args else "buffered"
            opened = re.match(r"(\d+)", result)
            if opened:
                descriptors[int(opened[1])] = channel
        marker = bisect.bisect_right(boundaries, time) - 1
        preparation = bisect.bisect_right(preparations, (time, len(trials))) - 1
        if preparation >= 0 and (marker < 0 or preparations[preparation][0] > boundaries[marker]):
            trial_index, phase = preparations[preparation][1], "preparation"
        elif marker >= 0:
            phase = ("open", "search", "cleanup")[marker % 3]
            trial_index = marker // 3
        else:
            continue
        if "/proc/" not in args:
            trial = trials[trial_index]
            entry = trial.setdefault(phase, {}).setdefault(name + ":" + channel,
                                                          {"calls": 0, "seconds": 0., "max_seconds": 0.})
            entry["calls"] += 1
            entry["seconds"] += duration
            entry["max_seconds"] = max(entry["max_seconds"], duration)
        if name == "close" and fd:
            descriptors.pop(int(fd[1]), None)
    return trials


def run(main, before, candidate, output):
    output = Path(output).resolve()
    results = {}
    for version, directory in (("main", main), ("before", before), ("candidate", candidate)):
        for count in (1, 64):
            stem = f"cold-ip-{version}-n{count}"
            trace = output / (stem + ".trace")
            command = ["strace", "-f", "-ttt", "-T", "-yy", "-s", "128", "-o", str(trace),
                       "-e", "trace=link,linkat,unlink,unlinkat,openat,close,read,pread64,preadv,io_setup,io_submit,io_getevents,io_destroy,io_uring_setup,io_uring_enter,futex,mmap,munmap,madvise,fadvise64,fsync,clock_nanosleep",
                       str(Path(directory).resolve() / "vector_system_search"), "cold", str(output / "auto"),
                       "ip32", "bufferpool", "0", str(count), "5"]
            with (output / (stem + ".csv")).open("w") as out, (output / (stem + ".log")).open("w") as err:
                subprocess.run(command, stdout=out, stderr=err,
                               env=os.environ | {"VECTOR_PERF_ARCH": "auto"}, check=True)
            phases = summarize(trace.read_text().splitlines())
            rows = list(csv.DictReader((output / (stem + ".csv")).open()))
            if len(rows) != len(phases):
                raise ValueError("trace/CSV trial count mismatch")
            results[stem] = [{"benchmark": row, "syscalls": phase} for row, phase in zip(rows, phases)]
    (output / "cold-ip-syscall-phases.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    run(*sys.argv[1:])
