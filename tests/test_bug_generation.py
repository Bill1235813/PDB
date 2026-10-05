import dspy
import pytest

import bug_generation as bg
from utils import file_diff

GT = "\n".join([
    "def f(xs):",
    "    total = 0",
    "    for x in xs:",
    "        total += x",
    "    count = len(xs)",
    "    mean = total / count",
    "    scale = 2",
    "    out = mean * scale",
    "    out = out + 1",
    "    return out",
])


def _swap(code, lineno, new):
    lines = code.splitlines()
    lines[lineno - 1] = new
    return "\n".join(lines)


class FakeHandler:
    """BigCodeBench-like handler: list feedback, no n_workers in its own harness."""
    merge_adjacent_add_blocks = False

    def __init__(self, detected):
        self.detected = detected  # predicate on buggy code
        self.calls = []

    def save_formatted_gt(self, prefix, data):
        return prefix + ".jsonl"

    def build_verify_unit_test(self, prefix, results, sol_field="solution"):
        self.entries = [(r["task_id"], r[sol_field]) for r in results]
        return prefix + ".jsonl"

    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=20, timeout=1800, n_workers=None):
        self.calls.append(n_workers)
        fail = [t for t, code in self.entries if self.detected(code)]
        ok = [t for t, code in self.entries if not self.detected(code)]
        return fail, ok, ["details"] * len(fail)


def _task(**kw):
    return {"task_id": "T/1", "gt_solution": GT, "task_prompt": "p", "test": "TESTS", **kw}


def _mark(data):
    from dataset.base import DatasetHandler

    class H(DatasetHandler):
        preprocess = verify_unit_test = build_verify_unit_test = save_formatted_gt = lambda *a, **k: None
    H().mark_editable_lines(data)


def test_bug_generate_with_list_feedback_handler(tmp_path, monkeypatch):
    """Regression: BCB/LCB return list feedback and must not crash the pipeline."""
    buggy = _swap(GT, 7, "    scale = 3")

    class FakeInjector:
        def __call__(self, **kw):
            return dspy.Prediction(subtype="Assignment", buggy_code=buggy)

    monkeypatch.setattr(bg, "BugInjector", FakeInjector)
    data = [_task()]
    _mark(data)
    handler = FakeHandler(detected=lambda code: code != GT)
    _, verified = bg.bug_generate(data, handler, str(tmp_path / "log"), bug_per_example=1,
                                  n_workers_validation=7)
    assert handler.calls == [7]
    assert len(verified) == 1 and verified[0]["task_id"] == "T/1" and verified[0]["is_buggy"]
    assert "FAIL_TO_PASS" not in verified[0]


@pytest.mark.parametrize("full_max,expected", [(4, 6), (1, 3)])
def test_atomicity_probe_count(tmp_path, monkeypatch, full_max, expected):
    buggy = GT
    for ln, new in [(7, "    scale = 3"), (8, "    out = mean - scale"), (9, "    out = out - 1")]:
        buggy = _swap(buggy, ln, new)

    class FakeInjector:
        def __init__(self, max_lines_per_block=4):
            pass

        def __call__(self, **kw):
            return dspy.Prediction(subtype="Algorithm", buggy_code=buggy)

    monkeypatch.setattr(bg, "MultilineBugInjector", FakeInjector)
    data = [_task()]
    _mark(data)
    handler = FakeHandler(detected=lambda code: code != GT)
    bg.bug_generate(data, handler, str(tmp_path / "log"), bug_per_example=1, mode="multi",
                    max_lines_per_block=4, atomicity_full_max_lines=full_max)
    atoms = [t for t, _ in handler.entries if "__atom__" in t]
    assert len(atoms) == expected  # 2^3 - 2 partial repairs, or 3 single-line reverts


def _single_bug(line, new, **extra):
    buggy = _swap(GT, line, new)
    _, _, diff = file_diff(GT, buggy)
    return {"task_id": "T/1", "gt_solution": GT, "task_prompt": "p", "buggy_code": buggy, "diff": diff,
            "bug_count": 1, "bug_type": "Assignment", "bug_subtype": "s", "is_buggy": True,
            "frozen_lines": 0, "gt_length": 10, "editable_lines": 7, "deletable_lines": 5, **extra}


def test_bug_compose_keeps_task_fields_but_not_per_bug_fields():
    swe = {"repo": "r", "image_name": "i", "target_file": "a.py",
           "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": ["t2"]}
    singles = [_single_bug(2, "    total = 1", **swe), _single_bug(7, "    scale = 3", **swe),
               _single_bug(10, "    return mean", **swe)]
    out = bg.bug_compose(singles, max_bugs=2, compose_per_example=5, stride=2, max_lines_per_block=1)
    k1 = [d for d in out if d["bug_count"] == 1]
    k2 = [d for d in out if d["bug_count"] == 2]
    assert len(k1) == 3 and k2
    assert all(d["FAIL_TO_PASS"] == ["t1"] and d["bug_type"] == "Assignment" for d in k1)
    for d in k2:
        assert (d["repo"], d["image_name"], d["target_file"]) == ("r", "i", "a.py")
        assert d["test"] is None
        assert not {"bug_type", "bug_subtype", "is_buggy", "FAIL_TO_PASS", "editable_lines"} & set(d)
    assert len({d["task_id"] for d in out}) == len(out)


def test_bug_compose_carries_test_field_like_before():
    singles = [_single_bug(2, "    total = 1", test="TESTS"), _single_bug(7, "    scale = 3", test="TESTS")]
    out = bg.bug_compose(singles, max_bugs=2, compose_per_example=3, stride=2, max_lines_per_block=1)
    assert all(d["test"] == "TESTS" for d in out if d["bug_count"] == 2)


def test_validate_and_sample_handles_bug_count_above_max_bugs():
    singles = [_single_bug(2, "    total = 1")]
    for d in singles:
        d["gt_diff"] = file_diff(d["buggy_code"], d["gt_solution"], cleaned=True)[2]
        d["task_id"] = "T/1_0"
        d["bug_count"] = 5  # e.g. a span-mode bug whose edits form 5 blocks
    assert len(bg.validate_and_sample(singles, max_bugs=3, max_gen_per_bin=5)) == 1
