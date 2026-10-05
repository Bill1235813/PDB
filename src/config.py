"""
Central configuration constants for PDB.

Edit these values to change defaults across the pipeline (prompts, CLI defaults,
evaluator tolerance) without hunting through individual files.
"""

# --- Multiline bug parameters ---
# NOTE: [design thought] MIN/MAX multiline lines are used in:
#   - prompt text for IntroduceMultilineBug (module.py)
#   - action descriptions (examples.py)
#   - contiguous-range enumeration (bug_generation.py)
#   - CLI default for --max_lines_per_block (bug_generation.py)
# Keeping them here means one edit updates everything.
MIN_MULTILINES = 2
MAX_MULTILINES = 4

# --- Evaluator defaults ---
# Default `--tolerance` (extra predicted lines per matched GT block for full
# precision credit). `0` = strict, `1` = multiline, `2` = single-line default.
DEFAULT_TOLERANCE_MULTILINE = 1
DEFAULT_TOLERANCE_SINGLELINE = 2

# Upper bound used in the evaluator when validating GT diffs: we accept GT
# blocks up to this size without rejection. Should be >= the largest
# --max_lines_per_block used for generation (30 for SWE-smith).
EVAL_MAX_LINES_PER_BLOCK = 30
