import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import test_adequacy as ta

SRC = textwrap.dedent('''\
    """Module docstring."""
    import os


    def area(w, h):
        """Docstring."""
        # a comment
        if w < 0:
            raise ValueError(
                "negative width"
            )
        else:
            result = w * h
        return result


    def unused(x):
        y = x + 1
        return y
''')
# line numbers of interest
L_IF, L_RAISE, L_RAISE_ARG, L_ELSE, L_RESULT, L_RETURN = 8, 9, 10, 12, 13, 14
L_UNUSED_Y, L_UNUSED_RET, L_COMMENT, L_DOC = 18, 19, 7, 6


def test_statement_owners_classifies_lines():
    owners = ta.statement_owners(SRC)
    assert L_COMMENT not in owners and L_DOC not in owners and 1 not in owners
    assert owners[L_RAISE] == owners[L_RAISE_ARG]  # continuation line belongs to the raise
    assert owners[L_ELSE] == owners[L_IF]  # `else:` belongs to the if statement
    assert owners[L_RESULT] != owners[L_IF]


def test_covered_lines_by_statement():
    # Tracer saw: if-test, assignment, return; the raise never ran; unused() never called.
    executable, covered = ta.covered_lines(SRC, [L_IF, L_RESULT, L_RETURN])
    assert {L_IF, L_ELSE, L_RESULT, L_RETURN} <= covered
    assert not {L_RAISE, L_RAISE_ARG, L_UNUSED_Y, L_UNUSED_RET} & covered
    assert {L_RAISE, L_UNUSED_Y} <= executable


class FakeHandler:
    def __init__(self, hits, aug_hits=None, statuses=None):
        self.hits, self.aug_hits, self.statuses = hits, aug_hits or {}, list(statuses or [])
        self.runs = []

    def measure_line_coverage(self, tasks, extra_tests=None, extra_setup=""):
        src = self.aug_hits if extra_tests is not None else self.hits
        return {t["task_id"]: {"hit_lines": src[t["task_id"]]} for t in tasks if t["task_id"] in src}

    def run_tests_in_container(self, repo, files, test_ids, extra_setup=""):
        self.runs.append(dict(files))
        return self.statuses.pop(0)

    def augmented_test_path(self, task, index=0):
        return f"pdb_augmented_tests/test_m_{index}.py"

    def module_name(self, task):
        return "m"


def _task(tid="t"):
    lines = SRC.splitlines()
    editable = [(i, l) for i, l in enumerate(lines, 1) if l.strip()
                and not any(k in l for k in ("def", "import", "class"))]
    return {"task_id": tid, "repo": "r", "target_file": "m.py", "gt_solution": SRC,
            "editable_lines": editable, "deletable_lines": list(editable)}


def test_gate_excludes_unexecuted_lines():
    tasks = [_task("t"), _task("unmeasured")]
    ta.coverage_gate(tasks, FakeHandler({"t": [L_IF, L_RESULT, L_RETURN]}))
    t, u = tasks
    assert t["coverage"]["gate_passed"] is False
    assert {L_RAISE, L_UNUSED_Y} <= set(t["coverage"]["uncovered_editable_lines"])
    assert u["coverage"] == {"measured": False, "gate_passed": None}
    ta.apply_coverage_filter(tasks)
    kept = {ln for ln, _ in t["editable_lines"]}
    assert L_RESULT in kept and not {L_RAISE, L_COMMENT, L_DOC, L_UNUSED_Y} & kept
    assert len(u["editable_lines"]) == len(_task()["editable_lines"])  # untouched


def test_augmentation_keeps_only_tests_passing_on_gt_in_every_run(monkeypatch):
    path = "pdb_augmented_tests/test_m_0.py"
    monkeypatch.setattr(ta, "generate_tests", lambda task, h, n_files=1, property_based=False:
                        {path: "def test_neg(): ...\ndef test_flaky(): ...\ndef test_wrong(): ...\n"})
    statuses = [{f"{path}::test_neg": "PASSED", f"{path}::test_flaky": "PASSED", f"{path}::test_wrong": "FAILED"},
                {f"{path}::test_neg": "PASSED", f"{path}::test_flaky": "FAILED", f"{path}::test_wrong": "FAILED"}]
    h = FakeHandler({"t": [L_IF, L_RESULT, L_RETURN]}, aug_hits={"t": [L_RAISE]}, statuses=statuses)
    tasks = [_task()]
    ta.run_test_adequacy(tasks, h, augment=True, repeats=2)
    t = tasks[0]
    assert t["augmented_test_ids"] == [f"{path}::test_neg"]
    assert list(t["augmented_tests"]) == [path]
    assert h.runs[0]["m.py"] == SRC  # validated against the ground truth
    assert t["coverage"]["augmented"] is True
    assert L_RAISE in t["coverage"]["covered_lines"]  # newly covered by the kept test
    assert L_UNUSED_Y in t["coverage"]["uncovered_editable_lines"]
    assert L_RAISE in {ln for ln, _ in t["editable_lines"]}


def test_line_tracer_end_to_end(tmp_path):
    """Run the stdlib tracer on a real pytest session (what runs inside the container)."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "geom.py").write_text(SRC)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_geom.py").write_text(
        "from pkg.geom import area\n\ndef test_area():\n    assert area(2, 3) == 6\n")
    tracer = Path(__file__).resolve().parents[1] / "src" / "line_tracer.py"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path)] + sys.path)}
    subprocess.run([sys.executable, str(tracer), "--out", "hits.json", "--target", "pkg/geom.py",
                    "--", "-q", "-p", "no:cacheprovider", "tests"],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    hits = json.loads((tmp_path / "hits.json").read_text())["pkg/geom.py"]
    _, covered = ta.covered_lines(SRC, hits)
    assert {L_IF, L_ELSE, L_RESULT, L_RETURN} <= covered
    assert not {L_RAISE, L_UNUSED_Y, L_UNUSED_RET} & covered
