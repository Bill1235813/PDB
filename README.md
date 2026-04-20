# PDB: Precise Debugging Benchmarking

📄 Paper *(coming soon)* &nbsp;·&nbsp;
🌐 [Project page](https://precise-debugging-benchmark.github.io/) &nbsp;·&nbsp;
🤗 [Datasets](https://huggingface.co/Precise-Debugging-Benchmarking) &nbsp;·&nbsp;
🏆 [Leaderboard](https://precise-debugging-benchmark.github.io/leaderboard.html)

**PDB** is an automatic pipeline that turns any coding dataset into a *debugging* benchmark with fine-grained metrics. Beyond binary unit-test scores, PDB evaluates a debugger with **edit-level precision** (did the model touch only the lines it had to?) and **bug-level recall** (did it fix every fault?). This rewards targeted fixes and penalizes the regeneration behavior frontier LLMs often fall back on.

- Released datasets: [`PDB-Single`](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Single) · [`PDB-Single-Hard`](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Single-Hard) · [`PDB-Multi`](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Multi)

> TL;DR — Frontier models like GPT-5.1-Codex and DeepSeek-V3.2-Thinking top unit-test leaderboards (>76%) but score at or below 45% on precision: they pass tests by rewriting, not repairing. PDB makes that gap measurable.

---

## 📦 Installation

We use [`uv`](https://docs.astral.sh/uv/) for reproducible environments.

```bash
git clone https://github.com/Bill1235813/PDB
cd PDB
uv sync                        # creates .venv, installs locked deps
source .venv/bin/activate      # optional; scripts already point at .venv/bin/python
```

The LiveCodeBench and BigCodeBench sandboxes live in separate uv envs:

```bash
cd dataset/bigcodebench/install   && uv sync --extra eval && cd -
cd dataset/livecodebench/install  && uv sync              && cd -
```

### API keys

Drop one key file per provider into [keys/](keys/) (each file is a single line with the raw key). Mapping, local-model setup, and `--model_api_file` override instructions are in [keys/README.md](keys/README.md).

---

## 🧪 Evaluate a model on PDB (single / single-hard / multi)

Bug-correct + score one model across both BigCodeBench and LiveCodeBench:

```bash
bash scripts/simple_debug_eval.sh <subset> <model>
```

- `<subset>` ∈ `single`, `single-hard`, `multi` (points at `<bench>_pdb_<subset>.json`).
- `<model>` is any [dspy](https://dspy.ai/) model string: `openai/gpt-5.1-codex`, `anthropic/claude-sonnet-4-5-20250929`, `deepseek/deepseek-chat`, etc. Local / self-hosted endpoints are supported too — see [scripts/README.md](scripts/README.md#local--self-hosted-model-evaluation).

Example output (Evaluator per-dataset lines + driver union):

```
[summary] gpt-5.1-codex on bigcodebench_pdb_single_hard round 1: unit=0.733 prec=0.548 rec=0.777 f1=0.602 (n=2510)
[summary] gpt-5.1-codex on livecodebench_pdb_single_hard round 1: unit=0.914 prec=0.465 rec=0.789 f1=0.540 (n=3224)
  union  unit=0.828 prec=0.500 rec=0.783 f1=0.566 (n=5734)
```

To loop a fixed list of reference models instead of one, run [scripts/run_debug_eval.sh](scripts/run_debug_eval.sh) with the same subset arg. Model list, token budgets, and run-wide knobs (debug mode, rounds, temperature) are configurable at the top of each driver — see [scripts/README.md](scripts/README.md) for details.

---

## 📐 Score an existing debug-results file

If you already have patches saved (downloaded from Hugging Face, produced by an external agent, etc.), score them without re-running the model:

```bash
python src/evaluator.py \
  --dataset_name bigcodebench \
  --eval_model_name my-model \
  --input_file my-model_on_bigcodebench_pdb_single_round_1.json \
  --eval_set_name bigcodebench_pdb_single \
  --max_iter 1
```

The input format matches what `bug_correct.py` writes (a list of entries with `task_id`, `buggy_code`, `gt_solution`, `debug_results.solution`, `gt_diff`, …). At the end, the same `[summary]` line as above is printed. Output-file paths and schema are documented in [scripts/README.md](scripts/README.md#output-files).

---

## 🐛 Generate your own PDB test set

Every file under `dataset/<bench>/data/full_data.json` goes through the same pipeline:

```bash
python src/bug_generation.py \
  --dataset_name bigcodebench \
  --model_name openai/gpt-5.1-codex \
  --input_file full_data.json \
  --output_prefix oai_buggy_code \
  --mode single          \   # or --mode multi
  --stride 2             \   # 2 for single, 4 for multi
  --max_lines_per_block 1 \  # 1 for single, 2-4 for multi
  --max_bugs 4            \  # max composed block count (bug_count)
  --bug_per_time 20       \  # per-task LLM call budget
  --max_gen_per_bin 5     \  # subsampling cap
  --temperature 1.0 --max_tokens 32000
```

The generator produces `oai_buggy_code_<timestamp>.json` under `results/<bench>/bug_data/`. Three-model fan-out drivers:

- [scripts/run_bug_gen_single.sh](scripts/run_bug_gen_single.sh) — single-line (3 models × 2 datasets, then merge into `<bench>_pdb_single.json`)
- [scripts/run_bug_gen_multi.sh](scripts/run_bug_gen_multi.sh) — multi-line (reads per-model `long_<name>.json` splits, merges into `<bench>_pdb_multi.json`)

Both scripts preflight API keys against a cheap probe before spending credits, and they run all 6 (model × dataset) jobs concurrently.

### Choose your generator pool

The default pool is GPT-5.1-Codex + Claude-4.5-Sonnet + Gemini-2.5-Pro. Swap the `MODELS` array in `run_bug_gen_*.sh` to taste — anything supported by LiteLLM works.

### Add a new source dataset

Implement a `DatasetHandler` subclass under `dataset/<your-dataset>/` and register it in `dataset/__init__.py`. See [dataset/README.md](dataset/README.md) for the full interface + vendored-sandbox layout. The rest of the pipeline is dataset-agnostic once the handler exists.

### Parameters reference

| flag | default | meaning |
|---|---|---|
| `--mode` | `single` | `single` (1-line bugs) or `multi` (contiguous 2-4 line blocks) |
| `--stride` | `2` | minimum inter-block line gap during composition (`s`) |
| `--max_bugs` | `4` | `k_max` — max bugs composed into a single program |
| `--bug_per_time` | `20` | `m_1` — LLM calls per `(x, C_gt)` pair for atomic-bug drafting |
| `--max_gen_per_bin` | `5` | `m_3` — subsample cap per `(task, bug_count)` bin |
| `--max_lines_per_block` | 1 single / 4 multi | block size cap for diff validation |
| `--temperature` | `1.0` | sampling temperature |
| `--max_tokens` | `32000` | thinking-budget cap |

---

## 🔁 Iterative or agentic debugging

All three flavors below start from an already-scored round-1 single-pass run (produced by `scripts/run_debug_eval.sh` or `simple_debug_eval.sh`) and reload it with `--reload_first_round`, so only rounds 2+ consume fresh API credits.

### 5.1 Iterative (text-only feedback)

The debugger sees its prior failed patches appended to `failed_attempts`, and the template auto-switches to `*_with_feedback` between rounds. No unit-test content or error traces are exposed.

```bash
python src/bug_correct.py \
  --dataset_name bigcodebench \
  --input_file bigcodebench_pdb_single_hard.json \
  --eval_set_name bigcodebench_pdb_single_hard \
  --model_name openai/gpt-5.1-codex \
  --debug_mode minimal --max_rounds 3 \
  --reload_first_round \
  --reload_result_file results/bigcodebench/debug_results/gpt-5.1-codex_on_bigcodebench_pdb_single_hard_round_1.json \
  --reload_score_file  results/bigcodebench/eval_results/gpt-5.1-codex_on_bigcodebench_pdb_single_hard_round_1_scores.json \
  --temperature 1.0 --max_tokens 32000
```

### 5.2 Agentic (tests + error messages exposed)

Same as iterative, but `--use_tests` puts the hidden unit tests into the prompt and `--error_msg` injects the sandbox's stdout/stderr for every failing attempt:

```bash
python src/bug_correct.py \
  --dataset_name bigcodebench \
  --input_file bigcodebench_pdb_single_hard.json \
  --eval_set_name bigcodebench_pdb_single_hard_agentic \
  --model_name openai/gpt-5.1-codex \
  --debug_mode minimal --max_rounds 3 \
  --use_tests --error_msg \
  --reload_first_round \
  --reload_result_file results/bigcodebench/debug_results/gpt-5.1-codex_on_bigcodebench_pdb_single_hard_round_1.json \
  --reload_score_file  results/bigcodebench/eval_results/gpt-5.1-codex_on_bigcodebench_pdb_single_hard_round_1_scores.json \
  --temperature 1.0 --max_tokens 32000
```

### 5.3 Agentic with Claude Code (tool-using subagent)

Swaps the single-pass dspy LM for an autonomous [Claude Code](https://claude.com/claude-code) subagent that can read the buggy code, execute tests, and iteratively patch. Routed through [src/claude_code_wrapper.py](src/claude_code_wrapper.py).

```bash
python src/bug_correct.py \
  --dataset_name bigcodebench \
  --input_file bigcodebench_pdb_single_hard.json \
  --eval_set_name bigcodebench_pdb_single_hard_claudecode \
  --model_name claude-code-agent \
  --use_claude_code \
  --debug_mode minimal --max_rounds 1 \
  --timeout 300
```

`--use_claude_code` implies `--use_tests` internally (the agent is expected to run them), so unit tests are always available. `--max_rounds 1` is typical here because the agent already iterates within its own loop.

Each round of every flavor writes `<model>_on_<eval_set>_round_<k>.json` + its scores file. Prompt templates (`minimal` vs `free`, `*_with_feedback`, `*_unit`) are documented in [scripts/README.md](scripts/README.md#prompt-variants).

---

## 🔬 Reproduce experiments

### Regenerate `<bench>_pdb_single.json`

```bash
bash scripts/run_bug_gen_single.sh          # 3 generators × 2 datasets + merge
```

### Build `<bench>_pdb_single_hard.json`

After scoring the 9 reference models on `_pdb_single`, filter to tasks solved perfectly by < 7 of 9:

```bash
bash scripts/run_debug_eval.sh single       # populates eval_results/
# then run the hard-filter cell in visualize/visualize.ipynb, or use the
# self-contained routine at the bottom of scripts/push_to_hf.py which follows
# the same logic.
```

This is exactly the procedure that produces the **PDB-Single-Hard** release set (5,734 examples).

### Regenerate `<bench>_pdb_multi.json`

Requires the `dataset/<bench>/data/long_<name>.json` per-generator splits of tasks with ≥ 35-line canonical solutions.

```bash
bash scripts/run_bug_gen_multi.sh
```

### Full 9-model evaluation

```bash
bash scripts/run_debug_eval.sh single-hard
bash scripts/run_debug_eval.sh multi
```

Final reproduction targets (union over BCB + LCB):

| subset | n | models evaluated | top precision model | top unit-score model |
|---|---|---|---|---|
| PDB-Single | 7,591 | 9 | Claude-Sonnet-4.5 | 	DeepSeek-V3.2-Thinking |
| PDB-Single-Hard | 5,751 | 9 | Claude-Sonnet-4.5 | 	DeepSeek-V3.2-Thinking |
| PDB-Multi | 256 | 9 | Claude-Sonnet-4.5 | 	DeepSeek-V3.2-Thinking |

---

## Citation

```
@inproceedings{zhu2026pdb,
  title     = {Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?},
  author    = {Zhu, Wang Bill and Chai, Miaosen and Wang, Shangshang and Liu, Yejia and
               Bian, Song and Dong, Honghua and Neiswanger, Willie and Jia, Robin},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
}
```
