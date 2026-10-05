import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset import get_handler
from dataset.swesmith import handler as hmod
from dataset.swesmith.handler import (SWESmithHandler, _compute_fix_patch, _validation_id,
                                      MODE_BUG_GEN, MODE_FIX_EVAL)

REPO = "pallets__click.fde47b4b"
GT = "def f(x):\n    return x + 1\n"
BUGGY = "def f(x):\n    return x - 1\n"


def _task(task_id=f"{REPO}.src__click__utils_0", **kw):
    return {"task_id": task_id, "gt_solution": GT, "repo": REPO, "image_name": "img",
            "target_file": "src/click/utils.py", "task_prompt": "p", **kw}


class FakeProfile:
    min_pregold = False
    test_cmd = "source /opt/miniconda3/bin/activate; conda activate testbed; pytest --disable-warnings --color=no --tb=no --verbose"
    repo_name = REPO

    @staticmethod
    def log_parser(log):
        out = {}
        for line in log.splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = parts[1]
        return out


def _fake_sw(tmp_path, status="0_f2p", patched_log=None, ref_log=None, report=None):
    log_dir = tmp_path / "logs"
    registry = SimpleNamespace(get=lambda repo: FakeProfile())
    calls = []

    def run_validation(instance):
        calls.append(instance)
        folder = log_dir / instance["repo"] / instance["instance_id"]
        folder.mkdir(parents=True, exist_ok=True)
        if patched_log is not None:
            (folder / "test_output.txt").write_text(patched_log)
        if report is not None:
            (folder / "report.json").write_text(json.dumps(report))
        return {"status": status}

    if ref_log is not None:
        ref = log_dir / REPO / f"{REPO}.ref"
        ref.mkdir(parents=True)
        (ref / "test_output.txt").write_text(ref_log)
    return dict(run_validation=run_validation, run_patch_in_container=None, KEY_PATCH="patch",
                REF_SUFFIX=".ref", LOG_DIR=log_dir, LOG_TEST_OUTPUT_PRE_GOLD="test_output_pre_gold.txt",
                registry=registry, KEY_INSTANCE_ID="instance_id", LOG_REPORT="report.json",
                LOG_TEST_OUTPUT="test_output.txt", close_logger=lambda l: None), calls


def test_registered_and_lazy():
    h = get_handler("swesmith")
    assert isinstance(h, SWESmithHandler)
    assert h.merge_adjacent_add_blocks is True
    assert SWESmithHandler()._data_index is None  # nothing loaded at construction


def test_validation_id_satisfies_swesmith_instance_id_rule():
    vid = _validation_id(f"{REPO}.src__click__utils_3__atom__1_0", "patch text")
    assert vid.rsplit(".", 1)[0] == REPO  # swesmith RepoProfile.get_test_cmd assertion
    assert vid != _validation_id(f"{REPO}.src__click__utils_3__atom__1_0", "other patch")


def test_compute_fix_patch_roundtrip():
    patch = _compute_fix_patch(GT, BUGGY, "src/click/utils.py")
    assert patch.startswith("--- a/src/click/utils.py")
    assert "-    return x + 1" in patch and "+    return x - 1" in patch
    assert _compute_fix_patch(GT, GT, "x.py") == ""


def test_build_verify_modes_and_metadata(tmp_path):
    h = SWESmithHandler()
    parent = _task(buggy_code=BUGGY)
    atom = {"task_id": parent["task_id"] + "__atom__1_0", "buggy_code": GT, "solution": GT}
    h.save_formatted_gt(str(tmp_path / "gt"), [parent, atom])
    f = h.build_verify_unit_test(str(tmp_path / "v"), [parent, atom], sol_field="buggy_code")
    rows = [json.loads(l) for l in open(f)]
    assert [r["mode"] for r in rows] == [MODE_BUG_GEN] * 2
    assert rows[1]["repo"] == REPO and rows[1]["gt_solution"] == GT  # atom inherits parent

    # fix-eval: the Evaluator passes only task_id + solution; metadata comes from the index
    f = h.build_verify_unit_test(str(tmp_path / "e"), [{"task_id": parent["task_id"], "solution": BUGGY}])
    row = json.loads(open(f).readline())
    assert row["mode"] == MODE_FIX_EVAL and row["code"] == BUGGY and row["target_file"] == "src/click/utils.py"


def test_fix_eval_rejects_invalid_python_without_docker(tmp_path):
    h = SWESmithHandler()
    sw, calls = _fake_sw(tmp_path)
    entry = {**_task(), "code": "def f(x:\n    return"}
    assert h._judge_fix(entry, sw) == ("fail", "solution is not valid Python")
    assert calls == []


def test_fix_eval_missing_tests_count_as_failures(tmp_path):
    """swesmith reports 0_f2p when the patch breaks collection; PDB must not."""
    h = SWESmithHandler()
    ref = "tests/test_a.py::t1 PASSED\ntests/test_a.py::t2 PASSED\ntests/test_b.py::t3 FAILED\n"
    sw, _ = _fake_sw(tmp_path, status="0_f2p", patched_log="tests/test_a.py::t1 PASSED\n", ref_log=ref)
    status, info = h._judge_fix({**_task(), "code": BUGGY}, sw)
    assert status == "fail" and "t2" in info


def test_fix_eval_correct_when_all_clean_passing_tests_pass(tmp_path):
    h = SWESmithHandler()
    ref = "tests/test_a.py::t1 PASSED\ntests/test_b.py::t3 FAILED\n"
    sw, calls = _fake_sw(tmp_path, status="0_f2p", patched_log="tests/test_a.py::t1 PASSED\n", ref_log=ref)
    assert h._judge_fix({**_task(), "code": BUGGY}, sw) == ("correct", None)
    assert calls[0]["instance_id"].rsplit(".", 1)[0] == REPO
    # identical code needs no container run
    assert h._judge_fix({**_task(), "code": GT}, sw) == ("correct", None)


def test_bug_gen_judgement(tmp_path):
    h = SWESmithHandler()
    sw, _ = _fake_sw(tmp_path, status="1+_f2p", report={"FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": ["t2"]})
    assert h._judge_bug({**_task(), "code": BUGGY}, sw) == ("fail", {"FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": ["t2"]})
    sw, _ = _fake_sw(tmp_path / "b", status="0_f2p")
    assert h._judge_bug({**_task(), "code": BUGGY}, sw) == ("correct", None)
    sw, _ = _fake_sw(tmp_path / "c", status="timeout")
    assert h._judge_bug({**_task(), "code": BUGGY}, sw)[0] == "drop"


def test_bug_gen_uses_augmented_tests_when_suite_misses_the_bug(tmp_path, monkeypatch):
    h = SWESmithHandler()
    sw, _ = _fake_sw(tmp_path, status="0_f2p")
    monkeypatch.setattr(h, "run_tests_in_container",
                        lambda repo, files, ids, sw=None, extra_setup="": {ids[0]: "FAILED"})
    entry = {**_task(), "code": BUGGY, "augmented_test_ids": ["pdb_augmented_tests/test_x_0.py::t"],
             "augmented_tests": {"pdb_augmented_tests/test_x_0.py": "def t(): pass\n"}}
    status, info = h._judge_bug(entry, sw)
    assert status == "fail" and info["FAIL_TO_PASS"] == ["pdb_augmented_tests/test_x_0.py::t"]


def test_verify_unit_test_fix_eval_counts_errors_as_failures_and_resumes(tmp_path, monkeypatch):
    h = SWESmithHandler()
    sw, _ = _fake_sw(tmp_path)
    monkeypatch.setattr(hmod, "_import_swesmith", lambda: sw)
    monkeypatch.setattr(hmod, "_run_pregold", lambda *a, **k: None)
    tasks = [_task(f"{REPO}.f_{i}") for i in range(3)]
    h.save_formatted_gt(str(tmp_path / "gt"), tasks)
    vf = h.build_verify_unit_test(str(tmp_path / "v"),
                                  [{"task_id": t["task_id"], "solution": BUGGY} for t in tasks])
    seen = []

    def judge(entry, _sw):
        seen.append(entry["task_id"])
        if entry["task_id"].endswith("_2"):
            raise RuntimeError("docker down")
        return ("correct", None) if entry["task_id"].endswith("_0") else ("fail", "x")

    monkeypatch.setattr(h, "_judge_fix", judge)
    fail, correct, fb = h.verify_unit_test(vf, n_workers=2)
    assert correct == [f"{REPO}.f_0"] and sorted(fail) == [f"{REPO}.f_1", f"{REPO}.f_2"]
    assert "docker down" in fb[f"{REPO}.f_2"]


def test_pytest_args_and_container_command():
    h = SWESmithHandler()
    sw = {"registry": SimpleNamespace(get=lambda r: FakeProfile())}
    assert h._pytest_args(REPO, sw) == ["--disable-warnings", "--color=no", "--tb=no", "--verbose"]
    cmd = h._container_command(REPO, sw, "pytest x")
    assert cmd.startswith("source /opt/miniconda3/bin/activate; conda activate testbed; cd /testbed; pytest x")


def test_module_name_and_test_path():
    h = SWESmithHandler()
    t = _task()
    assert h.module_name(t) == "click.utils"
    assert h.module_name({"target_file": "tinydb/__init__.py"}) == "tinydb"
    assert h.augmented_test_path(t, 1) == "pdb_augmented_tests/test_src__click__utils_1.py"


def test_fix_eval_checkpoint_is_keyed_by_code(tmp_path, monkeypatch):
    """Re-evaluating different outputs under the same verify-file name re-runs them."""
    h = SWESmithHandler()
    sw, _ = _fake_sw(tmp_path)
    monkeypatch.setattr(hmod, "_import_swesmith", lambda: sw)
    monkeypatch.setattr(hmod, "_run_pregold", lambda *a, **k: None)
    t = _task(f"{REPO}.f_0")
    h.save_formatted_gt(str(tmp_path / "gt"), [t])
    runs = []

    def judge(entry, _sw):
        runs.append(entry["code"])
        if len(runs) == 1:
            raise RuntimeError("transient docker error")
        return ("correct", None)

    monkeypatch.setattr(h, "_judge_fix", judge)
    vf = h.build_verify_unit_test(str(tmp_path / "v"), [{"task_id": t["task_id"], "solution": BUGGY}])
    assert h.verify_unit_test(vf)[0] == [t["task_id"]]          # error -> scored as failure
    assert h.verify_unit_test(vf)[1] == [t["task_id"]]          # not cached -> retried
    vf = h.build_verify_unit_test(str(tmp_path / "v"), [{"task_id": t["task_id"], "solution": GT + "\n# x"}])
    h.verify_unit_test(vf)
    assert runs == [BUGGY, BUGGY, GT + "\n# x"]


def test_derived_probe_ids_resolve_to_their_task(tmp_path):
    """Evaluator equivalence/redundancy probes and atomicity probes need task metadata."""
    h = SWESmithHandler()
    tasks = [_task(f"{REPO}.src__click__utils_1"), _task(f"{REPO}.src__click__utils_12", target_file="b.py")]
    h.save_formatted_gt(str(tmp_path / "gt"), tasks)
    probes = [f"{REPO}.src__click__utils_1_0", f"{REPO}.src__click__utils_1_0_1_2",
              f"{REPO}.src__click__utils_12_3", f"{REPO}.src__click__utils_12__atom__1_0"]
    f = h.build_verify_unit_test(str(tmp_path / "v"), [{"task_id": p, "solution": GT} for p in probes])
    rows = {r["task_id"]: r for r in map(json.loads, open(f))}
    assert [rows[p]["target_file"] for p in probes] == ["src/click/utils.py"] * 2 + ["b.py"] * 2
    assert all(rows[p]["gt_solution"] == GT and rows[p]["repo"] == REPO for p in probes)


def test_identical_patch_reuses_existing_report(tmp_path):
    h = SWESmithHandler()
    sw, calls = _fake_sw(tmp_path, status="1+_f2p", report={"FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []})
    entry = {**_task(), "code": BUGGY}
    assert h._judge_bug(entry, sw)[0] == "fail"
    assert h._judge_bug(entry, sw)[0] == "fail"
    assert len(calls) == 1


def test_missing_test_check_respects_min_testing_selection(tmp_path):
    """Profiles with min_testing run only related test files; others are not 'missing'."""
    h = SWESmithHandler()
    ref = ("tests/test_a.py::t1 PASSED\ntests/test_a.py::t2 PASSED\n"
           "tests/test_b.py::t3 PASSED\ndocs/faq.rst::line:1 PASSED\n")
    sw, calls = _fake_sw(tmp_path, status="0_f2p", patched_log="tests/test_a.py::t1 PASSED\ntests/test_a.py::t2 PASSED\n",
                         ref_log=ref)
    run = sw["run_validation"]

    def run_with_eval_sh(instance):
        out = run(instance)
        (sw["LOG_DIR"] / instance["repo"] / instance["instance_id"] / "eval.sh").write_text(
            "+ pytest --disable-warnings --color=no --tb=no --verbose tests/test_a.py\n")
        return out

    sw["run_validation"] = run_with_eval_sh
    assert h._judge_fix({**_task(), "code": BUGGY}, sw) == ("correct", None)
    # a selected test that disappears is still a failure
    sw2, _ = _fake_sw(tmp_path / "x", status="0_f2p", patched_log="tests/test_a.py::t1 PASSED\n", ref_log=ref)
    run2 = sw2["run_validation"]
    sw2["run_validation"] = lambda inst: (run2(inst), (sw2["LOG_DIR"] / inst["repo"] / inst["instance_id"] / "eval.sh")
                                          .write_text("pytest -v tests/test_a.py\n"))[0]
    assert h._judge_fix({**_task(), "code": BUGGY}, sw2)[0] == "fail"
