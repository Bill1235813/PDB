"""
Build the released SWE-smith split of PDB-Wild from the raw generation output.

    python scripts/prepare_swesmith_release.py \
        --input  <raw swe_multiline.json or gen_data_multiline_244.json> \
        --output results/swesmith/bug_data/swesmith_pdb_multi.json

Fixes applied to the raw merged generation output:
  * duplicate task ids -- the raw file merges several generation runs whose
    per-task `_<n>` suffixes restart at 0, so distinct bugs share an id (and the
    evaluator, which keys results by task_id, would score only one of them).
    The first occurrence keeps its id; later ones get the next free suffix.
  * bug_count is recomputed as the number of gt_diff blocks (SWE-smith block rule).
  * per-bug labels (bug_type, bug_subtype, is_buggy) are dropped from entries with
    bug_count > 1, matching the BigCodeBench/LiveCodeBench releases; for composed
    entries the raw labels described only one of the composed bugs.
  * FAIL_TO_PASS / PASS_TO_PASS are dropped (scoring re-runs the full suite).
  * source_model is added, as in the other PDB releases.

An id map (position, old id, new id) is written next to the output so debug
results produced on the raw file (same order) can be re-keyed.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import parse_diff_to_blocks, file_diff, apply_diff  # noqa: E402

DROP_ALWAYS = ("FAIL_TO_PASS", "PASS_TO_PASS")
PER_BUG_LABELS = ("bug_type", "bug_subtype", "is_buggy")


def dedupe_ids(data):
    """Return [(old_id, new_id)] and rename duplicates in place."""
    used = {d["task_id"] for d in data}
    seen, mapping = set(), []
    for d in data:
        old = d["task_id"]
        if old in seen:
            base = old.rsplit("_", 1)[0]
            n = 0
            while f"{base}_{n}" in used:
                n += 1
            d["task_id"] = f"{base}_{n}"
            used.add(d["task_id"])
        seen.add(d["task_id"])
        mapping.append((old, d["task_id"]))
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/swesmith/bug_data/swesmith_pdb_multi.json")
    parser.add_argument("--source_model", default="claude-opus-4.7")
    args = parser.parse_args()

    data = json.load(open(args.input))
    mapping = dedupe_ids(data)
    renamed = sum(o != n for o, n in mapping)

    recounted = 0
    for d in data:
        for k in DROP_ALWAYS:
            d.pop(k, None)
        blocks = len(parse_diff_to_blocks(d["gt_diff"], merge_adjacent_add=True))
        if blocks != d["bug_count"]:
            recounted += 1
            d["bug_count"] = blocks
        if d["bug_count"] > 1:
            for k in PER_BUG_LABELS:
                d.pop(k, None)
        d["source_model"] = args.source_model
        # round-trip invariant used by validate_and_sample
        assert not file_diff(apply_diff(d["buggy_code"], d["gt_diff"]), d["gt_solution"])[2], d["task_id"]

    assert len({d["task_id"] for d in data}) == len(data)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    id_map = os.path.splitext(args.output)[0] + "_id_map.json"
    with open(id_map, "w") as f:
        json.dump([{"index": i, "old_task_id": o, "task_id": n} for i, (o, n) in enumerate(mapping)],
                  f, indent=1)

    print(f"{len(data)} examples -> {args.output}")
    print(f"  renamed duplicate ids: {renamed}; recomputed bug_count: {recounted}")
    print(f"  bug_count: {sorted(Counter(d['bug_count'] for d in data).items())}")
    print(f"  repos: {len({d['repo'] for d in data})}, files: "
          f"{len({(d['repo'], d['target_file']) for d in data})}")
    print(f"  id map -> {id_map}")


if __name__ == "__main__":
    main()
