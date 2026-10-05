"""
Bug correction (debugging) pipeline for PDB (Precise Debugging Benchmarking).

Given buggy code + task prompts, drives an LLM debugger through one or more
rounds of fix attempts, evaluates each round, and accumulates failed-attempt
feedback for the next round.

Round structure (see `eval_main`):
    - Round 1 calls `bug_correct` on raw buggy_data (or optionally reloads a
      previous round's results via --reload_first_round).
    - After each round, the Evaluator scores every attempt. Tasks that failed
      have their solution appended to `failed_attempts` (plus optional error
      messages) and their `debug_results` stripped so the next call to
      `bug_correct` re-attempts them.
    - Starting with round 2 the debug_mode is suffixed with "_with_feedback"
      so the Debugger signature consumes the accumulated failures.

Tolerance defaults (`DEFAULT_TOLERANCE_SINGLELINE`/`_MULTILINE` in config.py)
are picked based on --mode so scoring is fair to the bug granularity used.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import copy
import itertools
import json
import threading
import time
import dspy
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import file_diff
from module import Debugger
from evaluator import Evaluator
from api_config import resolve_api_key
from config import DEFAULT_TOLERANCE_MULTILINE, DEFAULT_TOLERANCE_SINGLELINE


def _fix_one(item, debugger, args, rd):
    """Call the Debugger on one item and return the log entry."""
    log_entry = copy.deepcopy(item)
    # NOTE: [design thought] items that kept `debug_results` from a prior
    # round are already-solved carry-overs; skip so we don't re-spend tokens.
    if "debug_results" in log_entry:
        return log_entry
    log_entry["round"] = rd
    task_id = log_entry.get("task_id")
    buggy_code = log_entry.get("buggy_code")
    task_prompt = log_entry.get("task_prompt")
    unit_tests_code = log_entry.get("test", None)
    failed_attempts = log_entry.get("failed_attempts", None)
    try:
        response = debugger(task_prompt=task_prompt, buggy_code=buggy_code, test_cases=unit_tests_code,
                            failures=failed_attempts, mode=args.debug_mode)
        raw_output = response.solution or ""
        # NOTE: [pedagogical] pred_diff direction matches gt_diff (buggy -> fix),
        # so the evaluator can compare predicted blocks against ground-truth blocks
        # using the same coordinate frame.
        log_entry["debug_results"] = {
            "model": args.model_name,
            "solution": raw_output,
            "pred_diff": file_diff(buggy_code, raw_output, cleaned=True)[2]
        }
    except Exception as e:
        # NOTE: [edge case callout] on API/parse failure we leave `debug_results`
        # unset so bug_correct retries the item; items that exhaust their retries
        # get an empty solution and count as a (failed) attempt.
        print(f"Error processing task_id {task_id}: {e}")
    return log_entry


def _empty_result(item, args):
    item["debug_results"] = {
        "model": args.model_name,
        "solution": "",
        "pred_diff": file_diff(item.get("buggy_code", ""), "", cleaned=True)[2]
    }
    return item


def bug_correct(data, debugger, output_file, args, rd):
    """Run one debugging round: call the Debugger on each still-unsolved item.

    NOTE: [performance improvement] When --n_workers > 1 we fan out the
    per-item Debugger calls with a ThreadPoolExecutor (I/O-bound LLM calls).
    dspy + litellm are thread-safe via the module-level settings; the
    evaluator stage still runs serially since it shells out to the dataset
    sandbox.

    Finished items are appended to `output_file + ".ckpt.jsonl"`; an interrupted
    run resumes from it. Items whose API call failed are retried up to
    --max_api_retries times with exponential backoff.
    """
    if not data:
        print("No buggy data to correct; skipping correction phase.")
        return []

    ckpt_file = output_file + ".ckpt.jsonl"
    done = {}
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done[item["task_id"]] = item
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut off by the interruption
        print(f"Resuming from checkpoint: {len(done)}/{len(data)} items already done")

    results = [done.get(item.get("task_id")) for item in data]
    ckpt_lock = threading.Lock()
    n_workers = getattr(args, "n_workers", 1)

    def _run(indices, items):
        def _finish(i, entry):
            results[i] = entry
            if "debug_results" in entry:
                with ckpt_lock, open(ckpt_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        if n_workers <= 1:
            for i in tqdm.tqdm(indices):
                _finish(i, _fix_one(items[i], debugger, args, rd))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_fix_one, items[i], debugger, args, rd): i for i in indices}
                for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    _finish(futures[fut], fut.result())

    _run([i for i, r in enumerate(results) if r is None], data)

    max_api_retries = getattr(args, "max_api_retries", 3)
    failed = [i for i, r in enumerate(results) if "debug_results" not in r]
    for attempt in range(1, max_api_retries + 1):
        if not failed:
            break
        wait = 30 * (2 ** (attempt - 1))  # 30s, 60s, 120s
        print(f"API retry {attempt}/{max_api_retries}: {len(failed)} items failed, retrying in {wait}s...")
        time.sleep(wait)
        _run(failed, results)
        failed = [i for i in failed if "debug_results" not in results[i]]
    for i in failed:
        print(f"Permanently failed after {max_api_retries} retries: {results[i].get('task_id')}")
        _empty_result(results[i], args)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)

    return results


def eval_main(args):
    """Round controller: load data/model -> loop rounds -> feed failures back."""
    data_dir = os.path.join("results", args.dataset_name, "bug_data")
    output_dir = os.path.join("results", args.dataset_name, "debug_results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not args.eval_set_name:
        args.eval_set_name = os.path.splitext(os.path.basename(args.input_file[0]))[0]
    output_prefix = args.model_name.split("/")[-1]
    if args.output_prefix:
        output_prefix = args.output_prefix + args.model_name.split("/")[-1]

    # Load the dataset
    if len(args.input_file) == 1:
        input_file = os.path.join(data_dir, args.input_file[0])
        buggy_data = json.load(open(input_file, "r"))
    else:
        input_files = [os.path.join(data_dir, args.input_file[i]) for i in range(len(args.input_file))]
        buggy_data = list(itertools.chain.from_iterable([json.load(open(in_file, "r")) for in_file in input_files]))
    # NOTE: [design thought] "_unit" suffix selects the Debugger signature variant
    # that also receives unit tests; required for Claude Code's agentic mode.
    if args.use_tests or args.use_claude_code:
        args.debug_mode += "_unit"
        for d in buggy_data:
            assert "test" in d, "Not having test in data but using unit-test-based debug mode!"

    # Load the model
    assert args.use_claude_code or args.model_name
    if args.skip_generation:
        debugger = None
    elif args.use_claude_code:
        print("Using Claude Code autonomous agent mode")
        from claude_code_wrapper import ClaudeCodeGenerator
        generator_cor = ClaudeCodeGenerator(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout
        )
        debugger = Debugger(model=generator_cor)
    else:
        print(f"Using model: {args.model_name}")
        api_key = resolve_api_key(args.model_name, args.model_api_file)
        if args.model_name.split("/")[0] == "together_ai":
            generator_cor = dspy.LM(args.model_name, api_key=api_key, api_base='https://api.together.xyz/v1',
                                    temperature=args.temperature, max_tokens=args.max_tokens, num_retries=3)
        else:
            lm_kwargs = dict(temperature=args.temperature, max_tokens=args.max_tokens, num_retries=3)
            if api_key is not None:  # None means ambient credentials (e.g. vertex_ai/ ADC)
                lm_kwargs["api_key"] = api_key
            if args.reasoning_effort:
                lm_kwargs["reasoning_effort"] = args.reasoning_effort
            generator_cor = dspy.LM(args.model_name, **lm_kwargs)
        dspy.settings.configure(lm=generator_cor)
        debugger = Debugger()

    # Dry-run: replace the LM with a mock so no API credit is consumed
    if getattr(args, 'dry_run', False) and not args.skip_generation:
        from unittest.mock import MagicMock
        mock_lm = MagicMock()
        mock_lm.return_value = ["```python\n# mock dry-run response\npass\n```"]
        dspy.settings.configure(lm=mock_lm)
        debugger = Debugger()
        print("DRY RUN: using mock LM, no API calls will be made")

    # init evaluator
    print(f"Init evaluator")
    if not args.eval_model_name:
        args.eval_model_name = args.model_name.split("/")[-1] if args.model_name else "claude_code"
    if not args.eval_set_name:
        args.eval_set_name = os.path.splitext(args.input_file[0])[0]
    evaluator = Evaluator(args)

    print(f"Enter debugging process")
    for rd in range(1, args.max_rounds + 1):
        print(f"Round {rd}")

        # NOTE: [design thought] reload path skips the (expensive) first-round
        # LLM call by loading prior results + scores verbatim, so we can iterate
        # on later rounds without re-running round 1.
        if rd == 1 and args.reload_first_round:
            results = json.load(open(args.reload_result_file))
            scores = json.load(open(args.reload_score_file))
            buggy_dict = {d["task_id"]: d for d in buggy_data}
            filtered_results = []
            for item in results:
                if item["task_id"] in buggy_dict:
                    if args.use_tests or args.use_claude_code:
                        item["test"] = buggy_dict[item["task_id"]]["test"]
                    filtered_results.append(item)
            results = filtered_results
            scores = {metric: {task_id: v for task_id, v in metric_dict.items() if task_id in buggy_dict} for
                      metric, metric_dict in scores.items()}
            evaluator.result_formatting(results)
            evaluator.scores = scores
            if args.error_msg:
                evaluator.unit_score("Unit score")
        else:
            # Run debugging process
            output_file = os.path.join(output_dir, output_prefix) + f"_on_{args.eval_set_name}_round_{rd}.json"
            if args.skip_generation:
                # Re-score saved outputs (e.g. after fixing a sandbox issue) without API calls.
                if not os.path.exists(output_file):
                    raise FileNotFoundError(f"--skip_generation: no saved results at {output_file}")
                results = json.load(open(output_file))
                print(f"Loaded {len(results)} saved results from {output_file}")
            else:
                results = bug_correct(buggy_data, debugger, output_file, args, rd)

            # Run evaluation and save outputs
            if not args.no_eval:
                evaluator.run_evaluation(results=results, round=rd)
            else:
                return

        # NOTE: [pedagogical] this is the round-to-round hand-off. For every
        # FAILED task we append the attempted solution (and optional error msg)
        # to `failed_attempts`, then DELETE `debug_results`. Deleting is what
        # marks the item as "still needs work" -- `bug_correct` skips items
        # that retain debug_results, so successes carry over untouched.
        buggy_data = results
        for d in buggy_data:
            if not evaluator.success_unit(d["task_id"]):
                failed_attempt = d["debug_results"]["solution"]
                if args.error_msg and evaluator.error_msg and evaluator.error_msg[d["task_id"]]:
                    failed_attempt = f"{failed_attempt}\nWith error message:\n{evaluator.error_msg[d['task_id']]}"
                if "failed_attempts" in d:
                    d["failed_attempts"] += "\nFailed attempt {}\n{}\n".format(rd, failed_attempt)
                else:
                    d["failed_attempts"] = "Failed attempt {}\n{}\n".format(rd, failed_attempt)
                del d["debug_results"]

        # NOTE: [design thought] switch the Debugger signature to the feedback
        # variant once we have failures to feed in. Checked with endswith so
        # compound suffixes like "minimal_unit_with_feedback" stay idempotent.
        if not args.debug_mode.endswith("_with_feedback"):
            args.debug_mode += "_with_feedback"
            print(f"Switching debug mode to {args.debug_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--input_file", nargs='+', help="Input buggy file path, under output/{dataset_name}",
                        required=True)
    parser.add_argument("--debug_mode", choices=["free", "minimal"], default="minimal", type=str)
    parser.add_argument("--use_tests", action="store_true", help="Whether to use test cases")
    parser.add_argument("--output_prefix", type=str, help="Output file path, under eval/{dataset_name}", default="")

    # Model arguments
    parser.add_argument("--model_name", type=str, help="Evaluation model name", default=None)
    parser.add_argument("--model_api_file", type=str, default=None,
                        help="Model API file under keys/ (optional, auto-resolved from model name)")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the generator")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for model execution (seconds)")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Parallel workers for the Debugger fix-loop. "
                             "Only the LLM step is parallelized; evaluation still runs serially.")

    # Eval results arguments
    parser.add_argument("--no_eval", action="store_true", help="No evaluation after correction if on")
    parser.add_argument("--eval_result_dir", type=str, default="results")
    parser.add_argument("--eval_model_name", type=str, default=None)
    parser.add_argument("--eval_set_name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=2, help="Minimum stride between bug diffs")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="Bug mode. Controls the --tolerance default when not set explicitly.")
    parser.add_argument("--tolerance", type=int, default=None,
                        help=f"Per-block tolerance: extra predicted lines per matched GT block. "
                             f"0=strict. If unset, defaults to {DEFAULT_TOLERANCE_SINGLELINE} in "
                             f"--mode single and {DEFAULT_TOLERANCE_MULTILINE} in --mode multi.")

    # Iterative arguments
    parser.add_argument("--max_rounds", type=int, default=1, help="Maximum number of debugging rounds")
    parser.add_argument("--reload_first_round", action="store_true", help="Whether to reload first round results")
    parser.add_argument("--reload_result_file", type=str, default=1, help="The result file to reload")
    parser.add_argument("--reload_score_file", type=str, default=1, help="The score file to reload")
    parser.add_argument("--error_msg", action="store_true", help="Whether to provide error message")

    # Claude Code specific arguments
    parser.add_argument("--use_claude_code", action="store_true", help="Use Claude Code agent")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Reasoning effort for models that support it (ignored otherwise)")
    parser.add_argument("--max_api_retries", type=int, default=3,
                        help="Retries (with exponential backoff) for items whose API call failed")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Do not query the model; re-evaluate the saved results of a previous run")

    # Testing
    parser.add_argument("--dry_run", action="store_true",
                        help="Replace LLM with a mock that returns dummy output (no API credit used)")

    args = parser.parse_args()
    # NOTE: [pedagogical] tolerance = extra predicted lines allowed per matched
    # GT block. Multi-line bugs justify larger tolerance because a correct fix
    # naturally touches more surrounding context than a single-line one.
    # Resolve --tolerance default from --mode if not explicitly passed.
    if args.tolerance is None:
        args.tolerance = (DEFAULT_TOLERANCE_MULTILINE if args.mode == "multi"
                          else DEFAULT_TOLERANCE_SINGLELINE)
    eval_main(args)
