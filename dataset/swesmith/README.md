# SWE-smith

Handler + vendored dependencies for repository-level PDB tasks built from [SWE-smith](https://github.com/SWE-bench/SWE-smith) Docker images. This is the repository-scale part of **PDB-Wild** (`results/swesmith/bug_data/swesmith_pdb_multi.json`, 228 examples from Claude-Opus-4.7-generated multi-line bugs).

## Overview

Each task is one Python file of a SWE-smith repository (`target_file`), with the file's clean content as `gt_solution`. Programs are validated by applying a unified diff to the clean repository inside the repo's Docker image and running its full test suite through `swesmith.harness.valid`. [handler.py](handler.py) (`SWESmithHandler`) works in two modes, recorded per entry in the verify file:

- **bug-gen** (`build_verify_unit_test(..., sol_field="buggy_code")`): a candidate bug is valid iff it breaks at least one test that passes on the clean repository.
- **fix-eval** (Evaluator): a model fix is correct iff every test that passes on the clean repository still passes. Tests that disappear from the run (e.g. the fix breaks collection with a syntax or import error) count as failures, and output that is not valid Python is rejected without starting a container.

Task metadata is taken from the entries passed to `save_formatted_gt` (both the generation pipeline and the Evaluator call it before verifying), so no separate index file is needed. Validation runs use the instance id `<task_id>__<patch hash>` for containers and logs, so concurrent evaluations of different models do not collide. Logs are cached under `logs/run_validation/<repo>/`.

Multi-line SWE-smith bugs can be code moves, which `file_diff` reports as an `Add` next to a `Delete`. The handler sets `merge_adjacent_add_blocks = True`, so these are parsed as one block (BigCodeBench/LiveCodeBench keep the original rule).

## Install

```bash
uv sync --extra swesmith      # docker SDK + swesmith/swebench runtime deps
```

The `swesmith` and `swebench` packages are vendored under [install/](install/) (see [install/README.md](install/README.md)) and put on `sys.path` by the handler. A running Docker daemon is required; images (`swebench/swesmith.x86_64.*`, x86_64 only) are pulled on first use. On Apple Silicon enable Rosetta emulation in Docker Desktop.

## Generate

```bash
bash scripts/run_bug_gen_swesmith.sh
```

Per repository this runs [src/gen_swesmith_data.py](../../src/gen_swesmith_data.py) (extract 3 files of 150–750 lines from the image) and `src/bug_generation.py --mode multi --multi_validation span --max_lines_per_block 30 --max_bugs 3`, then merges the per-repo outputs of the run. `dataset/swesmith/data/gen_swesmith.*.json` are the file extracts used for the released set.

### Optional test-adequacy gate

```bash
COVERAGE_GATE=1 AUGMENT_TESTS=1 [PROPERTY_BASED=1] bash scripts/run_bug_gen_swesmith.sh
# or, standalone (writes <input>_gated.json, consumed by bug_generation.py as usual):
python src/test_adequacy.py --dataset_name swesmith --input_file gen_<image>.json \
    [--augment --model_name openrouter/anthropic/claude-opus-4-7 [--property_based]]
```

- **Coverage gate**: the repo's suite is run once on the clean code under [src/line_tracer.py](../../src/line_tracer.py) (stdlib-only, copied into the container). An editable line is covered iff the statement it belongs to executed; uncovered lines (and comments/docstrings) are removed from `editable_lines`/`deletable_lines` and never offered to the bug injector. The verdict is stored per task in `coverage`.
- **Test augmentation**: for tasks that fail the gate, the LLM writes pytest (optionally Hypothesis) tests aimed at the uncovered lines. A test is kept only if it passes on the ground truth in every one of `--repeats` runs. Kept tests are stored as `augmented_tests` / `augmented_test_ids` and are run by the handler next to the original suite in both bug validation and fix scoring.

## Evaluate

```bash
python src/bug_correct.py --dataset_name swesmith --input_file swesmith_pdb_multi.json \
    --model_name <model> --mode multi --max_tokens 32000 --n_workers 8
# or as part of the multi subset:
WITH_SWESMITH=1 bash scripts/run_debug_eval.sh multi
```

## Layout

```
dataset/swesmith/
├── README.md
├── handler.py                       # SWESmithHandler
├── data/gen_swesmith.*.json         # per-repo file extracts (inputs to bug generation)
└── install/                         # vendored swesmith / swebench packages
```

## Troubleshooting

- **`ImportError: swesmith/swebench not importable`** — run `uv sync --extra swesmith`; check that `install/SWE-smith/swesmith` and `install/SWE-bench/swebench` exist.
- **`Docker daemon is not running`** / container errors — start Docker; during fix-eval an entry whose container could not run is scored as incorrect and the reason is kept in the feedback.
- **Pre-gold baseline times out** — some suites are slow; the clean baseline is cached under `logs/run_validation/<repo>/<repo>.ref/` and reused.
