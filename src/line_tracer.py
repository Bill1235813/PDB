"""
Minimal line-coverage tracer used by the optional coverage gate (test_adequacy.py).

Runs pytest in-process and records which lines of the --target files execute.
It depends only on the standard library so it can be copied into any test
environment (e.g. a SWE-smith Docker image) without installing coverage.py.

Usage:
    python line_tracer.py --out hits.json --target pkg/mod.py [--target ...] -- <pytest args>

Writes {"<target>": [executed line numbers], ...} to --out. pytest's exit code
is ignored: tests that already fail on the clean code simply cover nothing.
"""
import argparse
import json
import os
import sys
import threading


def _matcher(targets):
    """Map a code object's filename to the --target it belongs to (or None)."""
    norm = [(t, os.path.realpath(t), "/" + t.lstrip("./")) for t in targets]
    cache = {}

    def match(filename):
        if filename not in cache:
            real = os.path.realpath(filename)
            cache[filename] = next((t for t, rt, suffix in norm
                                    if real == rt or real.endswith(suffix)), None)
        return cache[filename]

    return match


def _install_settrace(match, hits):
    def local(frame, event, arg):
        if event == "line":
            hits[match(frame.f_code.co_filename)].add(frame.f_lineno)
        return local

    def global_trace(frame, event, arg):
        if match(frame.f_code.co_filename) is None:
            return None
        hits[match(frame.f_code.co_filename)].add(frame.f_lineno)
        return local

    threading.settrace(global_trace)
    sys.settrace(global_trace)


def _install_monitoring(match, hits):
    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "pdb_line_tracer")

    def on_line(code, line):
        target = match(code.co_filename)
        if target is None:
            return mon.DISABLE
        hits[target].add(line)
        return mon.DISABLE  # each (code, line) only needs to be seen once

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.LINE)


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", action="append", default=[])
    args = parser.parse_args(argv[:split])
    pytest_args = argv[split + 1:]

    hits = {t: set() for t in args.target}
    match = _matcher(args.target)
    if hasattr(sys, "monitoring"):
        _install_monitoring(match, hits)
    else:
        _install_settrace(match, hits)

    try:
        import pytest
        pytest.main(pytest_args)
    finally:
        sys.settrace(None)
        threading.settrace(None)
        with open(args.out, "w") as f:
            json.dump({t: sorted(v) for t, v in hits.items()}, f)


if __name__ == "__main__":
    main()
