"""
gen_swesmith_data.py — Extract Python files from a SWE-smith Docker image and
create PDB-format data items for bug injection.

Usage
-----
python src/gen_swesmith_data.py \
    --image swesmith.x86_64.pandas-dev__pandas.95280573 \
    --max_files 30 \
    --min_lines 150 \
    --max_lines 750 \
    --output dataset/swesmith/data/pdb_swe_input.json

The output JSON is consumed by bug_generation.py:

    python src/bug_generation.py \
        --dataset_name swesmith \
        --input_file pdb_swe_input.json \
        --model_name openrouter/anthropic/claude-opus-4-7 \
        --mode multi --multi_validation span --max_lines_per_block 30 \
        --bug_per_time 10 --max_bugs 3 \
        --dry_run   # remove for real run
"""
import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path

# Make dataset/ importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def image_to_repo_key(image_name: str) -> str:
    """
    Extract the SWE-smith registry key (repo_name) from a Docker image name.

    Docker Hub format:  "swebench/swesmith.x86_64.msiemens_1776_tinydb.10644a0e"
    Legacy/dataset fmt: "swesmith.x86_64.msiemens__tinydb.10644a0e"
    Both → registry key: "msiemens__tinydb.10644a0e"

    The registry uses "__" as org/repo separator; the Docker image uses "_1776_"
    to avoid naming issues (Docker doesn't allow "__" in some contexts).
    """
    # Strip optional "swebench/" org prefix
    name = image_name.split("/")[-1]
    # Strip "swesmith.x86_64." prefix
    name = name.removeprefix("swesmith.x86_64.")
    # Normalise separator: _1776_ → __
    name = name.replace("_1776_", "__")
    return name


DEFAULT_EXCLUDE_PATTERNS = ("test_", "_test.py", "/tests/", "/test/", "/testing/",
                            "setup.py", "conf.py", "conftest.py",
                            "/docs/", "/doc/", "/examples/", "/benchmarks/")


def is_excluded(rel_path: str, exclude_patterns=DEFAULT_EXCLUDE_PATTERNS) -> bool:
    """True if rel_path is a test/doc/build file (patterns match on '/' + rel_path)."""
    # NOTE: [edge case callout] prefix "/" so directory patterns such as "/tests/"
    # also match top-level directories ("tests/base.py").
    path = "/" + rel_path
    return any(pat in path for pat in exclude_patterns)


def list_testbed_python_files(
    container,
    min_lines: int = 150,
    max_lines: int = 750,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[tuple[str, int]]:
    """
    List Python files inside a Docker container's /testbed that meet size criteria.

    Returns (path relative to /testbed, line count) pairs sorted by path, so
    seeded sampling downstream is reproducible.
    """
    result = container.exec_run(
        ["bash", "-c", "find /testbed -name '*.py' -type f -not -path '*/.git/*' "
                       "-exec wc -l {} + | grep -v ' total$'"],
        user="root",
    )
    if result.exit_code != 0:
        return []

    candidates = []
    for line in result.output.decode("utf-8", errors="replace").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1].startswith("/testbed/"):
            continue
        try:
            line_count = int(parts[0])
        except ValueError:
            continue
        rel_path = parts[1][len("/testbed/"):]
        if is_excluded(rel_path, exclude_patterns):
            continue
        if min_lines <= line_count <= max_lines:
            candidates.append((rel_path, line_count))
    return sorted(candidates)


def get_file_content(container, rel_path: str) -> str | None:
    """Read the content of /testbed/{rel_path} from a running Docker container."""
    result = container.exec_run(
        f"cat /testbed/{rel_path}",
        user="root",
    )
    if result.exit_code != 0:
        return None
    return result.output.decode("utf-8", errors="replace")


def extract_task_prompt(code: str, rel_path: str) -> str:
    """
    Extract a task description from a Python file.
    Uses the module docstring if present, otherwise falls back to file path.
    """
    try:
        tree = ast.parse(code)
        if (tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            docstring = tree.body[0].value.value.strip()
            # Truncate to a reasonable length
            if len(docstring) > 400:
                docstring = docstring[:400].rsplit("\n", 1)[0] + "..."
            if docstring:
                return f"File: {rel_path}\n\n{docstring}"
    except SyntaxError:
        pass
    return f"Fix bugs in {rel_path}"


def create_data_item(
    image_name: str,
    repo_key: str,
    rel_path: str,
    code: str,
) -> dict:
    """
    Create one PDB-format data item from a Python file extracted from Docker.

    Fields required by bug_generation.py / SWESmithHandler.mark_editable_lines:
        task_id, gt_solution, task_prompt, repo, image_name, target_file
    """
    # Derive a stable task_id from image + file path
    safe_path = rel_path.replace("/", "__").replace(".py", "").replace(".", "_")
    task_id = f"{repo_key}.{safe_path}"

    return {
        "task_id": task_id,
        "gt_solution": code,
        "task_prompt": extract_task_prompt(code, rel_path),
        "repo": repo_key,           # registry key: "pandas-dev__pandas.95280573"
        "image_name": image_name,   # original image name for reference
        "target_file": rel_path,    # relative to /testbed/
    }


def extract_files_from_image(
    image_name: str,
    max_files: int = 30,
    min_lines: int = 150,
    max_lines: int = 750,
    seed: int = 42,
    prefer_files: list[str] | None = None,
    relax_prefer_window: bool = False,
) -> list[dict]:
    """
    Start a Docker container from image_name, extract Python files, and
    return a list of PDB data items.

    prefer_files: list of relative paths to prioritize (e.g. from SWE-smith dataset).
    They must satisfy the same [min_lines, max_lines] window as random files unless
    relax_prefer_window is set (50-3000 lines). Random files fill any remaining slots.
    """
    try:
        import docker
    except ImportError:
        raise ImportError("docker-py not installed. Run: pip install docker")

    repo_key = image_to_repo_key(image_name)
    client = docker.from_env()

    print(f"[gen_swesmith_data] Starting container from {image_name}...")
    container = client.containers.create(
        image=image_name,
        command="tail -f /dev/null",
        detach=True,
        platform="linux/x86_64",
    )
    container.start()
    print(f"[gen_swesmith_data] Container {container.short_id} started.")

    try:
        items = []
        seen_paths = set()

        # --- Step 1: extract preferred files first (relaxed line-count window) ---
        if prefer_files:
            print(f"[gen_swesmith_data] Trying {len(prefer_files)} preferred file(s)...")
            for rel_path in prefer_files:
                if len(items) >= max_files:
                    break
                wc = container.exec_run(f"wc -l /testbed/{rel_path}", user="root")
                if wc.exit_code != 0:
                    print(f"[gen_swesmith_data]   - {rel_path} (not found, skipping)")
                    continue
                try:
                    lc = int(wc.output.decode().split()[0])
                except (ValueError, IndexError):
                    continue
                lo, hi = (50, 3000) if relax_prefer_window else (min_lines, max_lines)
                if not (lo <= lc <= hi):
                    print(f"[gen_swesmith_data]   - {rel_path} ({lc} lines, outside [{lo}, {hi}])")
                    continue
                if is_excluded(rel_path):
                    print(f"[gen_swesmith_data]   - {rel_path} (test/doc file, skipping)")
                    continue
                code = get_file_content(container, rel_path)
                if code is None:
                    continue
                item = create_data_item(image_name, repo_key, rel_path, code)
                items.append(item)
                seen_paths.add(rel_path)
                print(f"[gen_swesmith_data]   + {rel_path} ({lc} lines) [preferred]")

        # --- Step 2: fill remaining slots from random candidates ---
        remaining = max_files - len(items)
        if remaining > 0:
            print("[gen_swesmith_data] Listing Python files for random fill...")
            candidates = list_testbed_python_files(
                container, min_lines=min_lines, max_lines=max_lines
            )
            candidates = [(p, lc) for p, lc in candidates if p not in seen_paths]
            print(f"[gen_swesmith_data] Found {len(candidates)} additional candidate files.")

            if len(candidates) > remaining:
                random.Random(seed).shuffle(candidates)
                candidates = candidates[:remaining]

            for rel_path, line_count in candidates:
                code = get_file_content(container, rel_path)
                if code is None:
                    print(f"[gen_swesmith_data] Could not read {rel_path}, skipping.")
                    continue
                item = create_data_item(image_name, repo_key, rel_path, code)
                items.append(item)
                print(f"[gen_swesmith_data]   + {rel_path} ({line_count} lines)")

        return items

    finally:
        print(f"[gen_swesmith_data] Stopping container {container.short_id}...")
        container.stop(timeout=5)
        container.remove(force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Python files from a SWE-smith Docker image for PDB bug injection."
    )
    parser.add_argument(
        "--image", required=True,
        help="SWE-smith Docker image name, e.g. swesmith.x86_64.pandas-dev__pandas.95280573",
    )
    parser.add_argument(
        "--output", default="dataset/swesmith/data/pdb_swe_input.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max_files", type=int, default=30,
        help="Maximum number of Python files to extract per image",
    )
    parser.add_argument(
        "--min_lines", type=int, default=150,
        help="Minimum number of lines in a file to include",
    )
    parser.add_argument(
        "--max_lines", type=int, default=750,
        help="Maximum number of lines in a file to include",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for file selection",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing output file instead of overwriting",
    )
    parser.add_argument(
        "--prefer_files", type=str, default=None,
        help="Comma-separated list of file paths (relative to /testbed) to prioritize",
    )
    parser.add_argument(
        "--relax_prefer_window", action="store_true",
        help="Accept preferred files of 50-3000 lines instead of [min_lines, max_lines]",
    )
    args = parser.parse_args()

    prefer = [f.strip() for f in args.prefer_files.split(",")] if args.prefer_files else None

    items = extract_files_from_image(
        image_name=args.image,
        max_files=args.max_files,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        seed=args.seed,
        prefer_files=prefer,
        relax_prefer_window=args.relax_prefer_window,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.append and output_path.exists():
        existing = json.load(open(output_path))
        # Deduplicate by task_id
        existing_ids = {d["task_id"] for d in existing}
        items = [d for d in items if d["task_id"] not in existing_ids]
        items = existing + items

    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\n[gen_swesmith_data] Saved {len(items)} items to {output_path}")


if __name__ == "__main__":
    main()
