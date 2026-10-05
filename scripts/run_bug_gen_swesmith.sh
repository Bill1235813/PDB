#!/usr/bin/env bash
# Generate the SWE-smith part of PDB-Wild (repository-level multi-line bugs).
#
# Usage:  bash scripts/run_bug_gen_swesmith.sh
#         COVERAGE_GATE=1 [AUGMENT_TESTS=1 [PROPERTY_BASED=1]] bash scripts/run_bug_gen_swesmith.sh
#
# Per repo (in parallel): extract MAX_FILES files of MIN_LINES..MAX_LINES lines
# from the SWE-smith Docker image -> inject multi-line bugs (span validation,
# blocks of up to 30 lines) -> validate in Docker -> compose up to MAX_BUGS bugs.
# The per-repo outputs of THIS run are merged into
#   results/swesmith/bug_data/swesmith_pdb_multi_<timestamp>.json
#
# Requires: a running Docker daemon, `uv sync --extra swesmith`,
#           keys/openrouter_key.txt (or change MODEL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
PYTHON="$REPO_ROOT/.venv/bin/python"

# --- Run-wide knobs (paper settings) ---
MODEL="openrouter/anthropic/claude-opus-4-7"
MAX_FILES=3
MIN_LINES=150
MAX_LINES=750
MAX_LINES_PER_BLOCK=30
BUG_PER_TIME=10
MAX_BUGS=3
STRIDE=2
N_WORKERS_LLM=4
N_WORKERS_VALIDATION=4
COVERAGE_GATE="${COVERAGE_GATE:-0}"
AUGMENT_TESTS="${AUGMENT_TESTS:-0}"
PROPERTY_BASED="${PROPERTY_BASED:-0}"

IMAGES=(
  "swebench/swesmith.x86_64.msiemens_1776_tinydb.10644a0e"
  "swebench/swesmith.x86_64.scrapy_1776_scrapy.35212ec5"
  "swebench/swesmith.x86_64.pallets_1776_click.fde47b4b"
  "swebench/swesmith.x86_64.marshmallow-code_1776_marshmallow.9716fc62"
  "swebench/swesmith.x86_64.python-jsonschema_1776_jsonschema.93e0caa5"
  "swebench/swesmith.x86_64.arrow-py_1776_arrow.1d70d009"
  "swebench/swesmith.x86_64.lepture_1776_mistune.bf54ef67"
  "swebench/swesmith.x86_64.tornadoweb_1776_tornado.d5ac65c1"
  "swebench/swesmith.x86_64.seperman_1776_deepdiff.ed252022"
  "swebench/swesmith.x86_64.mahmoud_1776_boltons.3bfcfdd0"
)

prefer_files() {
  case "$1" in
    *tinydb*)      echo "tinydb/table.py,tinydb/queries.py,tinydb/utils.py,tinydb/database.py" ;;
    *scrapy*)      echo "scrapy/commands/genspider.py,scrapy/core/engine.py,scrapy/commands/startproject.py,scrapy/exporters.py" ;;
    *click*)       echo "src/click/core.py,src/click/utils.py,src/click/shell_completion.py,src/click/types.py,src/click/parser.py" ;;
    *marshmallow*) echo "src/marshmallow/fields.py,src/marshmallow/validate.py,src/marshmallow/schema.py,src/marshmallow/utils.py" ;;
    *jsonschema*)  echo "jsonschema/validators.py,jsonschema/_keywords.py,jsonschema/_legacy_keywords.py,jsonschema/exceptions.py" ;;
    *arrow*)       echo "arrow/arrow.py,arrow/parser.py,arrow/util.py,arrow/factory.py" ;;
    *mistune*)     echo "src/mistune/block_parser.py,src/mistune/inline_parser.py,src/mistune/renderers/html.py,src/mistune/renderers/rst.py,src/mistune/plugins/footnotes.py" ;;
    *tornado*)     echo "tornado/web.py,tornado/iostream.py,tornado/template.py,tornado/locale.py,tornado/websocket.py" ;;
    *deepdiff*)    echo "deepdiff/diff.py,deepdiff/delta.py,deepdiff/model.py,deepdiff/deephash.py,deepdiff/helper.py" ;;
    *boltons*)     echo "boltons/setutils.py,boltons/dictutils.py,boltons/urlutils.py,boltons/cacheutils.py,boltons/ioutils.py" ;;
    *)             echo "" ;;
  esac
}

GATE_ARGS=()
[[ "$COVERAGE_GATE" == "1" ]] && GATE_ARGS+=(--coverage_gate)
[[ "$AUGMENT_TESTS" == "1" ]] && GATE_ARGS+=(--augment_tests)
[[ "$PROPERTY_BASED" == "1" ]] && GATE_ARGS+=(--property_based)

TS="$(date +%m%d-%H%M)"
LOG_DIR="results/swesmith/bug_data/log/run_${TS}"
mkdir -p "$LOG_DIR"
echo "SWE-smith bug generation: ${#IMAGES[@]} repos, model=$MODEL, logs in $LOG_DIR"

run_repo() {
  local image="$1" tag
  tag="$(basename "$image")"
  local input="gen_${tag}.json"
  {
    $PYTHON src/gen_swesmith_data.py --image "$image" \
      --max_files "$MAX_FILES" --min_lines "$MIN_LINES" --max_lines "$MAX_LINES" \
      --prefer_files "$(prefer_files "$tag")" \
      --output "dataset/swesmith/data/$input"
    $PYTHON src/bug_generation.py --dataset_name swesmith --input_file "$input" \
      --model_name "$MODEL" --mode multi --multi_validation span \
      --max_lines_per_block "$MAX_LINES_PER_BLOCK" --stride "$STRIDE" \
      --bug_per_time "$BUG_PER_TIME" --max_bugs "$MAX_BUGS" --max_tokens 32000 \
      --reasoning_effort high --n_workers_llm "$N_WORKERS_LLM" \
      --n_workers_validation "$N_WORKERS_VALIDATION" \
      --log_prefix "${tag}_${TS}" --output_prefix "run_${TS}_${tag}" "${GATE_ARGS[@]}"
  } > "$LOG_DIR/${tag}.log" 2>&1 && echo "  ✓ $tag" || echo "  ✗ $tag (see $LOG_DIR/${tag}.log)"
}

docker info > /dev/null 2>&1 || { echo "ERROR: Docker daemon is not running"; exit 1; }
for image in "${IMAGES[@]}"; do
  run_repo "$image" &
done
wait

MERGED="results/swesmith/bug_data/swesmith_pdb_multi_${TS}.json"
TS="$TS" MERGED="$MERGED" $PYTHON - <<'EOF'
import glob, json, os
from collections import Counter
files = sorted(glob.glob(f"results/swesmith/bug_data/run_{os.environ['TS']}_*.json"))
data = [d for f in files for d in json.load(open(f))]
dups = [k for k, v in Counter(d["task_id"] for d in data).items() if v > 1]
assert not dups, f"duplicate task ids across repos: {dups[:5]}"
json.dump(data, open(os.environ["MERGED"], "w"), indent=2)
print(f"Merged {len(data)} examples from {len(files)} repos -> {os.environ['MERGED']}")
print("bug_count:", sorted(Counter(d["bug_count"] for d in data).items()))
EOF
