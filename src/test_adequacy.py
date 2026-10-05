"""
Optional test-adequacy gate for PDB preprocessing (paper §3.3).

Coverage gate:
    Before bug injection, run the task's test suite under a line tracer and
    require every editable line of the ground-truth solution to be executed by
    at least one test. Lines that are never executed cannot host a detectable
    bug, so they are removed from `editable_lines` / `deletable_lines` and are
    never offered to the bug injector.

Test augmentation:
    For tasks that fail the gate, an LLM writes additional tests aimed at the
    unexecuted lines. A generated test is kept only if it passes on the
    ground-truth solution (in every one of `repeats` runs, to drop flaky tests);
    tests can optionally be property-based (Hypothesis). The kept tests are
    stored on the task (`augmented_tests`, `augmented_test_ids`) and are run by
    the dataset handler alongside the original suite during bug validation and
    fix scoring, i.e. they feed the standard PDB validation and scoring path.

The dataset-specific parts (running a suite under the tracer, running extra
tests) are handler hooks; see `SWESmithHandler.measure_line_coverage` and
`SWESmithHandler.run_tests_in_container`.

Standalone usage (writes an annotated copy of the input that bug_generation.py
consumes directly; the gate's line filtering is re-applied after
mark_editable_lines):

    python src/test_adequacy.py --dataset_name swesmith \
        --input_file gen_swesmith.x86_64.pallets_1776_click.fde47b4b.json \
        [--augment --model_name openrouter/anthropic/claude-opus-4-7 [--property_based]]

or pass `--coverage_gate [--augment_tests ...]` to bug_generation.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import ast
import io
import json
import re
import tokenize


# ── line classification ──────────────────────────────────────────────────────

def _docstring_nodes(tree):
    """Expr nodes that are module/class/function docstrings (not executed code)."""
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                nodes.add(id(body[0]))
    return nodes


def _code_lines(code):
    """Line numbers that contain at least one non-comment token."""
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return lines


def statement_owners(code):
    """
    Map each executable line to the id of the innermost statement that owns it.

    NOTE: [design thought] A tracer reports one line per executed bytecode
    position, not every physical line: continuation lines of a multi-line call,
    `else:` / `except ...:` headers and decorator lines may never show up even
    when the statement ran. We therefore judge coverage per statement: a line is
    covered iff some traced line belongs to the same statement. Compound
    statements own their header lines (and else/except/finally keyword lines);
    nested statements own their own ranges. Docstrings, comments and blank lines
    are not executable and get no owner.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return {}
    docstrings = _docstring_nodes(tree)
    code_lines = _code_lines(code)
    owner = {}
    stmts = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                start = min([child.lineno] + [d.lineno for d in getattr(child, "decorator_list", [])])
                stmts.append((id(child) in docstrings, start, child.end_lineno, len(stmts)))
            visit(child)

    visit(tree)
    # Pre-order: parents come first, so nested statements overwrite their ranges.
    for is_doc, start, end, sid in stmts:
        for ln in range(start, end + 1):
            if is_doc:
                owner.pop(ln, None)
            else:
                owner[ln] = sid
    return {ln: sid for ln, sid in owner.items() if ln in code_lines}


def covered_lines(code, hit_lines):
    """(executable line set, covered line set) for gt code given traced hit lines."""
    owners = statement_owners(code)
    hit_stmts = {owners[ln] for ln in hit_lines if ln in owners}
    return set(owners), {ln for ln, sid in owners.items() if sid in hit_stmts}


# ── gate ─────────────────────────────────────────────────────────────────────

def _editable_line_numbers(task):
    return [ln if isinstance(ln, int) else ln[0] for ln in task.get("editable_lines", [])]


def annotate_coverage(task, hit_lines):
    """Attach the gate verdict for one task (uses the current editable_lines)."""
    executable, covered = covered_lines(task["gt_solution"], hit_lines)
    editable = _editable_line_numbers(task)
    uncovered = sorted(ln for ln in editable if ln in executable and ln not in covered)
    task["coverage"] = {
        "measured": True,
        "covered_lines": sorted(covered),
        "uncovered_editable_lines": uncovered,
        "non_executable_editable_lines": sorted(ln for ln in editable if ln not in executable),
        "gate_passed": not uncovered,
    }
    return task["coverage"]


def apply_coverage_filter(data):
    """
    Restrict editable/deletable lines to lines executed by the (augmented) suite.

    Call after handler.mark_editable_lines(); a no-op for tasks without a
    measured coverage annotation. Returns the number of excluded lines.
    """
    excluded = 0
    for task in data:
        cov = task.get("coverage") or {}
        if not cov.get("measured"):
            continue
        keep = set(cov["covered_lines"])
        for field in ("editable_lines", "deletable_lines"):
            if isinstance(task.get(field), list):
                before = len(task[field])
                task[field] = [l for l in task[field] if (l if isinstance(l, int) else l[0]) in keep]
                if field == "editable_lines":
                    excluded += before - len(task[field])
    return excluded


def coverage_gate(data, handler):
    """Measure coverage on the clean code and annotate every task in place."""
    results = handler.measure_line_coverage(data)
    for task in data:
        if task["task_id"] in results:
            annotate_coverage(task, results[task["task_id"]]["hit_lines"])
        else:
            task["coverage"] = {"measured": False, "gate_passed": None}
    measured = [t for t in data if t["coverage"]["measured"]]
    passed = sum(t["coverage"]["gate_passed"] for t in measured)
    print(f"[coverage gate] measured {len(measured)}/{len(data)} tasks; "
          f"{passed} pass, {len(measured) - passed} have unexecuted editable lines")
    return data


# ── test augmentation ────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
PBT_SETUP = "python -c 'import hypothesis' 2>/dev/null || pip install -q hypothesis"


def _extract_code(text):
    m = _FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip() + "\n"


def _numbered(code, marked):
    return "\n".join(f"{'>>' if i in marked else '  '} {i:4d} {line}"
                     for i, line in enumerate(code.splitlines(), 1))


def generate_tests(task, handler, n_files=1, property_based=False):
    """Ask the configured dspy LM for test files targeting uncovered lines."""
    import dspy
    from module import GenerateUnitTests

    uncovered = set(task["coverage"]["uncovered_editable_lines"])
    gen = dspy.Predict(GenerateUnitTests)
    files = {}
    for i in range(n_files):
        response = gen(
            module_path=task["target_file"],
            import_name=handler.module_name(task),
            source_code=_numbered(task["gt_solution"], uncovered),
            uncovered_lines=", ".join(map(str, sorted(uncovered))),
            style=("Use Hypothesis property-based tests (@given) where an invariant can be "
                   "stated; plain pytest tests otherwise." if property_based else
                   "Use plain pytest test functions."),
        )
        files[handler.augmented_test_path(task, i)] = _extract_code(response.test_code)
    return files


def _test_ids_in(status, path):
    return sorted(t for t in status if t.split("::", 1)[0] == path)


def augment_task(task, handler, n_files=1, repeats=2, property_based=False):
    """Generate tests for one gate-failing task; keep those passing on gt every run."""
    files = generate_tests(task, handler, n_files=n_files, property_based=property_based)
    setup = PBT_SETUP if property_based else ""
    passing = None
    for _ in range(repeats):
        status = handler.run_tests_in_container(
            task["repo"], {task["target_file"]: task["gt_solution"], **files},
            list(files), extra_setup=setup)
        ok = {t for path in files for t in _test_ids_in(status, path) if status[t] in ("PASSED", "XFAIL")}
        passing = ok if passing is None else passing & ok
    kept = sorted(passing or [])
    kept_files = {p: c for p, c in files.items() if any(t.startswith(p + "::") for t in kept)}
    task["augmented_tests"] = kept_files
    task["augmented_test_ids"] = kept
    task["augmented_test_setup"] = setup
    print(f"[augment] {task['task_id']}: kept {len(kept)} generated tests")
    return task


def augment_tests(data, handler, n_files=1, repeats=2, property_based=False, n_workers=4):
    """Augment every measured task that fails the gate, then re-measure coverage."""
    from concurrent.futures import ThreadPoolExecutor

    failing = [t for t in data if (t.get("coverage") or {}).get("gate_passed") is False]
    print(f"[augment] {len(failing)} tasks fail the coverage gate")
    if not failing:
        return data
    with ThreadPoolExecutor(max_workers=max(1, min(n_workers, len(failing)))) as pool:
        list(pool.map(lambda t: augment_task(t, handler, n_files, repeats, property_based), failing))

    with_tests = [t for t in failing if t.get("augmented_test_ids")]
    extra = {t["task_id"]: t["augmented_tests"] for t in with_tests}
    setup = PBT_SETUP if property_based else ""
    results = handler.measure_line_coverage(with_tests, extra_tests=extra, extra_setup=setup)
    for task in with_tests:
        if task["task_id"] not in results:
            continue
        # Hits of the augmented suite = original covered lines + new hits.
        new_hits = set(results[task["task_id"]]["hit_lines"]) | set(task["coverage"]["covered_lines"])
        annotate_coverage(task, new_hits)
        task["coverage"]["augmented"] = True
    fixed = sum(t["coverage"]["gate_passed"] for t in with_tests)
    print(f"[augment] {fixed}/{len(failing)} tasks pass the gate after augmentation")
    return data


def run_test_adequacy(data, handler, augment=False, n_files=1, repeats=2,
                      property_based=False, n_workers=4):
    """Gate (+ optional augmentation). Expects handler.mark_editable_lines(data) first."""
    if not hasattr(handler, "measure_line_coverage"):
        raise NotImplementedError(
            f"{type(handler).__name__} does not implement measure_line_coverage(); "
            "the coverage gate is currently available for: swesmith")
    coverage_gate(data, handler)
    if augment:
        augment_tests(data, handler, n_files=n_files, repeats=repeats,
                      property_based=property_based, n_workers=n_workers)
    excluded = apply_coverage_filter(data)
    print(f"[coverage gate] excluded {excluded} unexecuted editable lines from injection")
    return data


def main():
    import dspy
    from dataset import get_handler
    from api_config import resolve_api_key

    parser = argparse.ArgumentParser(description="Optional coverage gate + test augmentation.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--input_file", required=True, help="under dataset/<name>/data")
    parser.add_argument("--output_file", default=None,
                        help="under dataset/<name>/data (default: <input>_gated.json)")
    parser.add_argument("--augment", action="store_true", help="LLM test augmentation")
    parser.add_argument("--model_name", default=None, help="LLM for test augmentation")
    parser.add_argument("--model_api_file", default=None)
    parser.add_argument("--property_based", action="store_true")
    parser.add_argument("--n_test_files", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2,
                        help="a generated test is kept only if it passes in every run")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    data_dir = os.path.join("dataset", args.dataset_name, "data")
    data = json.load(open(os.path.join(data_dir, args.input_file)))
    handler = get_handler(args.dataset_name)
    data = handler.preprocess(data)
    if args.augment:
        assert args.model_name, "--augment requires --model_name"
        api_key = resolve_api_key(args.model_name, args.model_api_file)
        lm_kwargs = dict(temperature=args.temperature, max_tokens=args.max_tokens)
        if api_key is not None:
            lm_kwargs["api_key"] = api_key
        dspy.settings.configure(lm=dspy.LM(args.model_name, **lm_kwargs))

    handler.mark_editable_lines(data)
    run_test_adequacy(data, handler, augment=args.augment, n_files=args.n_test_files,
                      repeats=args.repeats, property_based=args.property_based,
                      n_workers=args.n_workers)
    for task in data:  # recomputed by bug_generation.py; keep the file compact
        for field in ("editable_lines", "deletable_lines", "frozen_lines", "gt_length"):
            task.pop(field, None)
    out = args.output_file or os.path.splitext(args.input_file)[0] + "_gated.json"
    with open(os.path.join(data_dir, out), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} tasks to {os.path.join(data_dir, out)}")


if __name__ == "__main__":
    main()
