from utils import file_diff, parse_diff_to_blocks, set_block_merge_mode, verify_span_diff

GT = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
# Move "b = 2" below "c = 3": file_diff reports an Add next to a Delete.
MOVED = "a = 1\nc = 3\nb = 2\nd = 4\ne = 5\n"


def _move_diff():
    _, _, diff = file_diff(MOVED, GT, cleaned=True)
    return diff


def test_default_rule_is_unchanged_for_original_datasets():
    diff = {"3": {"type": "Delete", "original": "x", "modified": ""},
            "2": {"type": "Add", "original": "", "modified": "y"}}
    diff = dict(sorted(diff.items(), key=lambda kv: int(kv[0])))
    assert len(parse_diff_to_blocks(diff)) == 2
    assert len(parse_diff_to_blocks(diff, merge_adjacent_add=True)) == 1


def test_module_default_follows_set_block_merge_mode():
    diff = {"2": {"type": "Add", "original": "", "modified": "y"},
            "3": {"type": "Delete", "original": "x", "modified": ""}}
    set_block_merge_mode(True)
    assert len(parse_diff_to_blocks(diff)) == 1
    assert len(parse_diff_to_blocks(diff, merge_adjacent_add=False)) == 2


def test_code_move_diff_is_parsed_consistently():
    diff = _move_diff()
    assert diff
    assert len(parse_diff_to_blocks(diff, merge_adjacent_add=True)) <= len(parse_diff_to_blocks(diff))


def test_verify_span_diff():
    diff = {str(i): {"type": "Modify", "original": "a", "modified": "b"} for i in (10, 12, 15)}
    assert verify_span_diff(diff, max_span=6, min_changes=2, max_changes=5)[0]
    assert not verify_span_diff(diff, max_span=5)[0]  # span 6
    assert not verify_span_diff(diff, max_span=30, min_changes=4)[0]
    assert not verify_span_diff(diff, max_span=30, max_changes=2)[0]
    assert not verify_span_diff({}, max_span=30)[0]
