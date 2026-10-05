import json
from argparse import Namespace
from pathlib import Path

import pytest

from evaluator import Evaluator
from utils import apply_diff, file_diff, parse_diff_to_blocks

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "results" / "swesmith" / "bug_data" / "swesmith_pdb_multi.json"


def _evaluator(tmp_path, dataset="swesmith"):
    args = Namespace(dataset_name=dataset, eval_result_dir=str(tmp_path), eval_model_name="m",
                     eval_set_name="s", stride=2, tolerance=1)
    return Evaluator(args)


def _with_pred(d):
    return {**d, "round": 1, "debug_results": {"solution": d["buggy_code"], "pred_diff": {}}}


@pytest.fixture(scope="module")
def release():
    return json.load(open(RELEASE))


def test_release_ids_unique_and_schema(release):
    assert len(release) == 228
    assert len({d["task_id"] for d in release}) == len(release)
    required = {"task_id", "gt_solution", "task_prompt", "buggy_code", "gt_diff", "bug_count",
                "repo", "image_name", "target_file", "source_model"}
    for d in release:
        assert required <= set(d)
        assert d["task_id"].rsplit(".", 1)[0] == d["repo"]  # swesmith instance-id rule
        if d["bug_count"] > 1:
            assert not {"bug_type", "bug_subtype", "is_buggy"} & set(d)
        assert not {"FAIL_TO_PASS", "PASS_TO_PASS"} & set(d)


def test_release_roundtrip_and_block_counts(release):
    for d in release:
        assert not file_diff(apply_diff(d["buggy_code"], d["gt_diff"]), d["gt_solution"])[2]
        assert len(parse_diff_to_blocks(d["gt_diff"], merge_adjacent_add=True)) == d["bug_count"]


def test_evaluator_accepts_every_release_example(tmp_path, release):
    ev = _evaluator(tmp_path)
    ev.result_formatting([_with_pred(d) for d in release])
    assert len(ev.eval_ids) == len(release)


def test_evaluator_rejects_duplicate_task_ids(tmp_path, release):
    ev = _evaluator(tmp_path)
    dup = [_with_pred(release[0]), _with_pred(release[0])]
    with pytest.raises(ValueError, match="Duplicate task_id"):
        ev.result_formatting(dup)


def test_unit_score_pairs_feedback_with_the_right_task(tmp_path):
    ev = _evaluator(tmp_path, dataset="bigcodebench")

    class H:
        def build_verify_unit_test(self, *a, **k):
            return "v"

        def save_formatted_gt(self, *a, **k):
            return None

        def verify_unit_test(self, *a, **k):
            ids = [f"t{i}" for i in range(20)]
            return ids, [], [f"msg-{i}" for i in ids]

    ev.handler = H()
    ev.eval_ids = [f"t{i}" for i in range(20)]
    ev.pred = [""] * 20
    ev.eval_results = ev.results = [{}] * 20
    ev.unit_score("Unit score")
    assert all(ev.error_msg[i] == f"msg-{i}" for i in ev.eval_ids)


def test_swesmith_unit_score_through_evaluator(tmp_path, monkeypatch, release):
    """Evaluator order: build_verify_unit_test is called before save_formatted_gt."""
    from dataset.swesmith import handler as hmod
    from dataset.swesmith.handler import SWESmithHandler

    ev = _evaluator(tmp_path)
    ev.handler = h = SWESmithHandler()
    monkeypatch.setattr(hmod, "_import_swesmith", lambda: {"registry": None, "run_patch_in_container": None,
                                                          "KEY_INSTANCE_ID": "i", "REF_SUFFIX": ".ref",
                                                          "LOG_DIR": tmp_path, "close_logger": None})
    monkeypatch.setattr(hmod, "_set_arch", lambda *a: None)
    monkeypatch.setattr(hmod, "_run_pregold", lambda *a, **k: None)
    seen = {}

    def judge(entry, sw):
        seen[entry["task_id"]] = entry
        return ("correct", None) if entry["code"] == entry["gt_solution"] else ("fail", "x")

    monkeypatch.setattr(h, "_judge_fix", judge)
    items = [_with_pred(d) for d in release[:3]]
    items[0]["debug_results"] = {"solution": items[0]["gt_solution"], "pred_diff": {}}
    ev.result_formatting(items)
    ev.unit_score("Unit score")
    assert all(e["repo"] and e["gt_solution"] and e["target_file"] for e in seen.values())
    assert ev.scores["Unit score"] == {items[0]["task_id"]: 1, items[1]["task_id"]: 0, items[2]["task_id"]: 0}
