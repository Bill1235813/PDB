#!/bin/bash
# Bug correction + evaluation with 9 models on both datasets.
#
# Usage:  bash scripts/run_debug_eval.sh <subset>
#   <subset> ∈ {single, single-hard, multi}  (default: single)
#     single       -> results/<ds>/bug_data/<ds>_pdb_single.json
#     single-hard  -> results/<ds>/bug_data/<ds>_pdb_single_hard.json
#     multi        -> results/<ds>/bug_data/<ds>_pdb_multi.json
#
#   WITH_SWESMITH=1 bash scripts/run_debug_eval.sh multi
#     also evaluates the SWE-smith part of PDB-Wild
#     (results/swesmith/bug_data/swesmith_pdb_multi.json; needs Docker and
#     `uv sync --extra swesmith`, see dataset/swesmith/README.md).
#
# Thinking models (temperature=1.0, max_tokens=32000):
#   deepseek-reasoner, gemini-2.5-pro, gpt-5.1-codex,
#   claude-sonnet-4-5-20250929, grok-code-fast-1,
#   Kimi-K2-Thinking, Qwen3-Coder-480B-A35B-Instruct-FP8
#
# Non-thinking models (temperature=1.0, max_tokens=8000):
#   deepseek-chat, Kimi-K2-Instruct
set -e

SUBSET="${1:-single}"
case "$SUBSET" in
  single)      FILE_TAG="pdb_single" ;;
  single-hard) FILE_TAG="pdb_single_hard" ;;
  multi)       FILE_TAG="pdb_multi" ;;
  *)
    echo "ERROR: unknown subset '$SUBSET' (expected: single | single-hard | multi)"
    exit 2
    ;;
esac
echo "Subset: $SUBSET  (file tag: $FILE_TAG)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:$PYTHONPATH"
export TRANSFORMERS_VERBOSITY=error    # silence bcb sandbox's PyTorch-not-found banner
PYTHON="$REPO_ROOT/.venv/bin/python"

# --- Trap and zombie cleanup ---
cleanup() {
  echo ""
  echo "Caught interrupt, cleaning up..."
  local pids
  pids=$(jobs -p 2>/dev/null)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
  fi
  kill_zombies
  exit 1
}
trap cleanup INT TERM

kill_zombies() {
  USER_NAME="$(id -un)"
  for pattern in \
    "dataset/bigcodebench/install/.venv/bin/python.*bigcodebench\.evaluate" \
    "dataset/livecodebench/install/.venv/bin/python.*lcb_runner"; do
    if pgrep -u "$USER_NAME" -af "$pattern" > /dev/null 2>&1; then
      echo "Killing zombie processes matching: $pattern"
      pkill -u "$USER_NAME" -f "$pattern" || true
    fi
  done
}

# --- Run-wide knobs (edit here to change everything) ---
DEBUG_MODE=minimal
MAX_ROUNDS=1
TEMPERATURE=1.0
MAX_TOKENS_THINKING=32000
MAX_TOKENS_NON_THINKING=8000
N_WORKERS=4

# --- Debugger models ---
# Thinking models get $MAX_TOKENS_THINKING; non-thinking get $MAX_TOKENS_NON_THINKING.
THINKING_MODELS=(
  "deepseek/deepseek-reasoner"
  "gemini/gemini-2.5-pro"
  "openai/gpt-5.1-codex"
  "anthropic/claude-sonnet-4-5-20250929"
  "xai/grok-code-fast-1"
  "together_ai/Kimi-K2-Thinking"
  "together_ai/Qwen3-Coder-480B-A35B-Instruct-FP8"
)
NON_THINKING_MODELS=(
  "deepseek/deepseek-chat"
  "together_ai/Kimi-K2-Instruct"
)

DATASETS=("bigcodebench" "livecodebench")
if [[ "${WITH_SWESMITH:-0}" == "1" ]]; then
  if [[ "$SUBSET" != "multi" ]]; then
    echo "ERROR: SWE-smith is only part of the multi subset"
    exit 2
  fi
  DATASETS+=("swesmith")
fi

# --- Multi subset requires --mode multi so tolerance defaults to 1 ---
MODE_ARGS=()
if [[ "$SUBSET" == "multi" ]]; then
  MODE_ARGS=(--mode multi)
fi

for dataset in "${DATASETS[@]}"; do
  echo "============================================"
  echo "=== Bug Correction + Eval: $dataset ($SUBSET) ==="
  echo "============================================"

  input="${dataset}_${FILE_TAG}.json"
  eval_set="${dataset}_${FILE_TAG}"

  # Sanity check the input exists before kicking off a model loop.
  if [[ ! -f "results/$dataset/bug_data/$input" ]]; then
    echo "ERROR: missing input file results/$dataset/bug_data/$input"
    exit 1
  fi

  # --- Thinking models ---
  for model in "${THINKING_MODELS[@]}"; do
    echo ""
    echo "--- [thinking] $model on $dataset ---"

    $PYTHON src/bug_correct.py \
      --dataset_name "$dataset" \
      --model_name "$model" \
      --input_file "$input" \
      --debug_mode "$DEBUG_MODE" \
      --max_rounds "$MAX_ROUNDS" \
      --eval_set_name "$eval_set" \
      --temperature "$TEMPERATURE" \
      --max_tokens "$MAX_TOKENS_THINKING" \
      --n_workers "$N_WORKERS" \
      "${MODE_ARGS[@]}"

    kill_zombies
  done

  # --- Non-thinking models ---
  for model in "${NON_THINKING_MODELS[@]}"; do
    echo ""
    echo "--- [non-thinking] $model on $dataset ---"

    $PYTHON src/bug_correct.py \
      --dataset_name "$dataset" \
      --model_name "$model" \
      --input_file "$input" \
      --debug_mode "$DEBUG_MODE" \
      --max_rounds "$MAX_ROUNDS" \
      --eval_set_name "$eval_set" \
      --temperature "$TEMPERATURE" \
      --max_tokens "$MAX_TOKENS_NON_THINKING" \
      --n_workers "$N_WORKERS" \
      "${MODE_ARGS[@]}"

    kill_zombies
  done
done

echo ""
echo "============================================"
echo "=== Union Summary ($SUBSET) ==="
echo "============================================"

DATASETS="${DATASETS[*]}" FILE_TAG="$FILE_TAG" MAX_ROUNDS="$MAX_ROUNDS" $PYTHON -c "
import json, os, glob
from collections import defaultdict

# NOTE: [design thought] Per-dataset per-model lines already printed by
# Evaluator.print_summary() inside each bug_correct.py call. This block only
# aggregates the FINAL round across datasets into a union-per-model table.
file_tag = os.environ['FILE_TAG']
final_round = int(os.environ['MAX_ROUNDS'])
datasets = os.environ['DATASETS'].split()
union = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # model -> sums + total n

for dataset in datasets:
    pattern = f'results/{dataset}/eval_results/*_on_{dataset}_{file_tag}_round_{final_round}_scores.json'
    for sf in sorted(glob.glob(pattern)):
        model = os.path.basename(sf).split('_on_')[0]
        scores = json.load(open(sf))
        unit = scores.get('Unit score', {})
        sym = scores.get('Symbolic block scores', {})
        n = len(unit)
        if n == 0:
            continue
        acc = union[model]
        acc[0] += sum(unit.values())
        acc[1] += sum(v['precision'] for v in sym.values())
        acc[2] += sum(v['recall']    for v in sym.values())
        acc[3] += sum(v['f1']        for v in sym.values())
        acc[4] += n

print()
for model in sorted(union):
    u, p, r, f, n = union[model]
    print(f'  {model:44s} unit={u/n:.3f} prec={p/n:.3f} rec={r/n:.3f} f1={f/n:.3f} (n={n})')
"
