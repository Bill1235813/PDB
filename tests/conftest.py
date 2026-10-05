import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

import utils  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_block_merge_mode():
    """Tests must not leak the process-wide block-merge setting."""
    utils.set_block_merge_mode(False)
    yield
    utils.set_block_merge_mode(False)


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
