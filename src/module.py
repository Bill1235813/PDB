"""
DSPy modules and prompt signatures for PDB (Precise Debugging Benchmarking).

Defines the Debugger, BugInjector, and Rewriter modules that wrap LLM calls
via dspy.Predict. Prompt text is intentionally left unchanged.
"""
import dspy
import re

from config import MIN_MULTILINES, MAX_MULTILINES

CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
SIMPLE_CODE_BLOCK_REGEX = re.compile(r"```(.*?)```", re.DOTALL | re.IGNORECASE)
DIFF_STR_PATTERN = re.compile(r"^(\d+): (.*) --> (.*)$")


class ExternalModelWrapper:
    def __init__(self, model):
        self.model = model
        self.free_template = """Debug the given Python code that contains errors. Do NOT add any comments. The input consists of three parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- A set of unit tests for the problem.

Your response should include:
- A self-contained, corrected Python implementation.

PART 1: Problem Description
```text
{task_prompt}
```

PART 2: Buggy Code
```python
{buggy_code}
```

PART 3: Unit Tests (context only)
```python
{unit_tests_code}
```

Output format (follow *exactly*):
```python
[Corrected code here]
```
Corrected Code Output (use the format above):
"""
        self.minimal_template = """Debug the given Python code that contains errors. ONLY fix the bugs. Make minimal edits.
Do NOT generate a new solution based on the problem description.
Do NOT reformat lines that are already correct.
Do NOT edit or add any comments.

The input consists of three parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- A set of unit tests for the problem.

Your response should include:
- A self-contained, corrected Python implementation, only making minimal edits on the buggy code.

PART 1: Problem Description
```text
{task_prompt}
```

PART 2: Buggy Code
```python
{buggy_code}
```

PART 3: Unit Tests (context only)
```python
{unit_tests_code}
```

Output format (follow *exactly*):
```python
[Corrected code here]
```
Corrected Code Output (use the format above):
"""

    def __call__(self, problem_description, buggy_solution, unit_tests, debug_mode):
        if debug_mode.startswith("minimal"):
            prompt_text = self.minimal_template.format(
                task_prompt=problem_description,
                unit_tests_code=unit_tests,
                buggy_code=buggy_solution
            )
        elif debug_mode.startswith("free"):
            prompt_text = self.free_template.format(
                task_prompt=problem_description,
                unit_tests_code=unit_tests,
                buggy_code=buggy_solution
            )
        else:
            raise ValueError("Unknown debug mode for external wrapper: {}".format(debug_mode))
        response = self.model(prompt_text)
        if response and isinstance(response, list) and len(response) > 0:
            raw_output = response[0]
        elif hasattr(response, 'completions') and response.completions:
            raw_output = response.completions[0].content
        else:
            raise ValueError("Unexpected response format from the model.")
        return raw_output


class RewriteSolution(dspy.Signature):
    """Your task is to rewrite the solution code of a task with structural and stylistic perturbations.
You will be given two parts:
PART 1: A task description outlining the intended functionality.
PART 2: A correct solution to the task.

First, read both the task description and the provided solution to understand what the code is supposed to do.
Then, your task is to perform a deep rewriting (perturbation) of the solution code WITHOUT changing its functionality.
Strictly adhere to the following rules:
    - Do NOT change the starter code as given in the task description.
    - Do NOT add any new comments.
    - Do NOT make shallow redundancy edits, such as a = 1 + 2 - 1.
    - Do NOT condense the code to very short format.
    - NEVER rename variables to very short names (e.g., sum_production -> p), but you can give variables wrong names.
    - The rewritten code should resemble what human will write and should NOT be hard for human to read.
    - The rewritten code should be different enough from the original code.
You can probably use the following hints:
    - Change loop syntax (e.g., for to while or vice versa, if applicable)
    - Convert recursion to iteration or vice versa (if functionally equivalent)
    - Invert control flow where logical structure remains the same (e.g., replace `if not ...` with an inverted block)
    - Merge or flatten adjacent or nested if blocks.

Your response should ONLY contain the rewritten Python code, which is a different but correct solution to the task."""
    task_description = dspy.InputField(desc="The programming task description.")
    original_solution = dspy.InputField(desc="The original Python code solution.")
    rewritten_solution = dspy.OutputField(desc="A functionally equivalent but different Python code solution.")


class IntroduceBug(dspy.Signature):
    """Your task is to perform a deep analysis of a code snippet and intentionally introduce ONE bug.
You will be given two major components:
PART 1: A task description outlining the intended functionality.
PART 2: A solution to the task.
First, carefully read and understand both the task description and the provided solution.
Then, modify the solution by injecting a realistic programming error.
You will be asked to introduce one of the following bug types into the code: Assignment, Checking, Algorithm, Build/Package/Merge or Timing/Serialization.
You will be asked to perform exactly one action: Add one line, Delete one line or Modify one line.

Instructions for modifying the code:
- ONLY modify ONE selected line to induce a HARD bug to the task.
- Keep the other lines of the code solution EXACTLY the SAME.
- Do NOT add any new comments to the modified line.
- DO NOT introduce easy bugs such as referencing variable names before declaration, adding typos or commas.

You should output two things:
- The subtype of the introduced bug.
- A buggy code snippet with only a ONE line difference with the original code solution, and no comments on that modified line."""
    task_prompt = dspy.InputField(desc="The programming task description for context")
    correct_solution = dspy.InputField(desc="A correct Python code solution")
    bug_type = dspy.InputField(desc="The type of bug to add")
    action_on_lines = dspy.InputField(desc="The action and the lines to choose from")
    subtype = dspy.OutputField(desc="Subtype of the introduced bug")
    buggy_solution = dspy.OutputField(desc="Code with only ONE bug introduced, no comments on the modified line")


def make_multiline_bug_signature(min_lines: int, max_lines: int):
    """Factory: create an IntroduceMultilineBug Signature with dynamic min/max line counts."""

    class _Sig(dspy.Signature):
        task_prompt = dspy.InputField(desc="The programming task description for context")
        correct_solution = dspy.InputField(desc="A correct Python code solution")
        bug_type = dspy.InputField(desc="The type of bug to add")
        action_on_lines = dspy.InputField(desc="Contiguous line ranges to choose from and their code")
        subtype = dspy.OutputField(desc="Subtype of the introduced bug")
        buggy_solution = dspy.OutputField(
            desc=f"Full code with exactly ONE contiguous {min_lines}-{max_lines} line bug block")

    _Sig.__doc__ = f"""Your task is to introduce ONE realistic, human-authored bug that spans a contiguous
BLOCK of {min_lines} to {max_lines} CONSECUTIVE lines in a Python solution.

Think like a sleep-deprived engineer reviewing their own code — the bug should be a plausible
slip a real developer might commit, not an obviously synthetic corruption.

You will be given:
PART 1: A task description.
PART 2: A correct solution.
PART 3: A set of contiguous line RANGES you may choose from (e.g., "Lines 10-13").

Pick exactly ONE of the given ranges. Within that range, change {min_lines}-{max_lines} consecutive lines.

GOOD multiline bug patterns to draw inspiration from:
- Flip a loop/branch condition AND mis-adjust a dependent expression in the next line(s)
  (e.g., `for i in range(n):` -> `for i in range(n-1):` together with `x[i+1] = ...` -> `x[i] = ...`).
- Off-by-one in a bound AND a corresponding off-by-one in an index or slice on a following line.
- Negate a guard AND drop the compensating branch body (swap then/else behavior).
- Swap two related variables across consecutive assignments (group1 / group2 aliasing).
- Replace a correct function call pair where both calls have their argument order subtly wrong.
- Forget to update one side of a symmetric two-line update (update left, forget right).

You may BLEND bug categories when it is natural. A Checking flip on one line can cascade into
an Assignment correction on the next line — that is exactly what real human bugs look like.

CRITICAL RULES:
- Only change lines within ONE of the given contiguous ranges; keep every other line EXACTLY the same.
- All changed lines must be CONSECUTIVE (no gaps between modified lines).
- Number of changed lines: {min_lines} to {max_lines}.
- Every changed line must change RUNTIME BEHAVIOR. Comment-only edits, whitespace edits, or
  consistent variable renames do NOT count and will be rejected.
- Each changed line must be ESSENTIAL. If one edit could be reverted to the GT while the rest
  still cause a test failure, that edit was not essential — drop it and pick a block where
  every line genuinely contributes.
- DO NOT invent fake library APIs (e.g., `server.bind_address`, `server.listen_backlog`,
  `platform.linux_distribution` on Python 3.8+). Use real Python / stdlib / common-library names.
  If a real API doesn't exist, pick a different bug.
- DO NOT introduce trivial bugs: typos, syntax errors, undefined variables, missing imports,
  or deleting a for/if/while/with/try header without its body.
- Preserve indentation of surrounding code.
- Do NOT add comments to modified lines, AND do NOT replace an existing line with a comment.
- Do NOT delete the header of a control-flow block (if/for/while/with/try/except) while keeping
  the body — this creates unreachable or nonsensical code.

BAD BUG EXAMPLES — never produce output that matches these patterns:

(A) Deletion causing syntax errors:
       ORIG:
         if os.path.exists(plot_path):
             plt.savefig(plot_path)
         plt.close()
       BAD BUGGY:
             plt.savefig(plot_path)           <-- header deleted, structure broken
         plt.close()

(B) Fake APIs that don't exist: `server.bind_address(host, port)`,
    `server.listen_backlog(5)`, `msg.set_content(subject)` when msg is an email.MIMEText,
    `platform.linux_distribution()` on Python 3.8+.

(C) Non-atomic edits: a block where reverting any single line to the GT would still leave
    the tests failing. That means only the non-reverted lines caused the failure, and the
    reverted line was superfluous.

Output:
- The subtype label for the introduced bug (if mixed, pick the dominant one or "Others").
- The full buggy code with exactly one contiguous block of {min_lines}-{max_lines} lines changed."""

    return _Sig


# Default signature using config constants (used when max_lines_per_block not specified)
IntroduceMultilineBug = make_multiline_bug_signature(MIN_MULTILINES, MAX_MULTILINES)



class MinimalDebug(dspy.Signature):
    """Debug the given Python code that contains errors. ONLY fix the bugs. Make minimal edits.
Do NOT generate a new solution based on the problem description.
Do NOT reformat lines that are already correct.
Do NOT edit or add any comments.

The input consists of two parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.

Your response should include:
- A self-contained, corrected Python implementation, with minimal edits."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class MinimalFeedbackDebug(dspy.Signature):
    """Debug the given Python code that contains errors. ONLY fix the bugs. Make minimal edits.
Do NOT generate a new solution based on the problem description.
Do NOT reformat lines that are already correct.
Do NOT edit or add any comments.

The input consists of three parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- Previously failed attempts and optionally error feedback.

Your response should include:
- A self-contained, corrected Python implementation, only making minimal edits on the buggy code."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    failed_attempts = dspy.InputField(desc="Previous attempts that failed unit tests, AVOID THEM!")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class MinimalUnitDebug(dspy.Signature):
    """Debug the given Python code that contains errors. ONLY fix the bugs. Make minimal edits.
Do NOT generate a new solution based on the problem description.
Do NOT reformat lines that are already correct.
Do NOT edit or add any comments.

The input consists of three parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- A set of unit tests for the problem.

Your response should include:
- A self-contained, corrected Python implementation, only making minimal edits on the buggy code."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    unit_tests = dspy.InputField(desc="The unit tests")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class MinimalUnitFeedbackDebug(dspy.Signature):
    """Debug the given Python code that contains errors. ONLY fix the bugs. Make minimal edits.
Do NOT generate a new solution based on the problem description.
Do NOT reformat lines that are already correct.
Do NOT edit or add any comments.

The input consists of four parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- A set of unit tests for the problem.
- Previously failed attempts and optionally error feedback.

Your response should include:
- A self-contained, corrected Python implementation, only making minimal edits on the buggy code."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    unit_tests = dspy.InputField(desc="The unit tests")
    failed_attempts = dspy.InputField(desc="Previous attempts that failed unit tests, AVOID THEM!")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class FreeDebug(dspy.Signature):
    """Debug the given Python code that contains errors. Do NOT add any comments. The input consists of two parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.

Your response should include:
- A self-contained, corrected Python implementation."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class FreeUnitDebug(dspy.Signature):
    """Debug the given Python code that contains errors. Do NOT add any comments. The input consists of three parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- A set of unit tests for the problem.

Your response should include:
- A self-contained, corrected Python implementation."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    unit_tests = dspy.InputField(desc="The unit tests")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class FreeFeedbackDebug(dspy.Signature):
    """Debug the given Python code that contains errors. Do NOT add any comments. The input consists of three parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- Previously failed attempts and optionally error feedback.

Your response should include:
- A self-contained, corrected Python implementation."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    failed_attempts = dspy.InputField(desc="Previous attempts that failed unit tests, AVOID THEM!")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class FreeUnitFeedbackDebug(dspy.Signature):
    """Debug the given Python code that contains errors. Do NOT add any comments. The input consists of four parts:
- A problem description outlining the intended functionality.
- A buggy code that needs to be fixed.
- A set of unit tests for the problem.
- Previously failed attempts and optionally error feedback.

Your response should include:
- A self-contained, corrected Python implementation."""
    problem_description = dspy.InputField(desc="The problem description")
    buggy_solution = dspy.InputField(desc="The buggy solution")
    unit_tests = dspy.InputField(desc="The unit tests")
    failed_attempts = dspy.InputField(desc="Previous attempts that failed unit tests, AVOID THEM!")
    corrected_solution = dspy.OutputField(desc="The corrected solution")


class GenerateUnitTests(dspy.Signature):
    """Write additional pytest tests for a Python module from a real repository.

The tests must exercise the lines marked with ">>" in the numbered source (they are not
executed by the existing test suite). The tests are run against the CORRECT code shown here
and are discarded if they fail, so assert only behavior that the shown code actually has.

RULES:
- Output one self-contained pytest file in a ```python code block.
- Import the module under test by its import name; do not copy its code into the test.
- Use only the standard library, pytest, the module's package and its existing dependencies
  (plus `hypothesis` if property-based tests are requested).
- Tests must be deterministic: no network, no sleeping, no reliance on wall-clock time;
  write files only under pytest's `tmp_path`.
- Prefer several small focused test functions over one large test."""
    module_path = dspy.InputField(desc="Path of the module inside the repository")
    import_name = dspy.InputField(desc="Dotted import name of the module")
    source_code = dspy.InputField(desc="Numbered source; '>>' marks lines the suite never executes")
    uncovered_lines = dspy.InputField(desc="Line numbers that new tests should execute")
    style = dspy.InputField(desc="Test style requirements")
    test_code = dspy.OutputField(desc="A complete pytest file in a ```python code block")


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteSolution)

    def forward(self, task_prompt, gt_solution):
        prediction = self.rewrite(task_description=task_prompt, original_solution=gt_solution)

        if prediction.rewritten_solution:
            match = CODE_BLOCK_REGEX.search(prediction.rewritten_solution)
            match_simple = SIMPLE_CODE_BLOCK_REGEX.search(prediction.rewritten_solution)
            if match:
                rewritten_solution = match.group(1).strip()
            elif match_simple:
                rewritten_solution = match_simple.group(1).strip()
            else:
                rewritten_solution = prediction.rewritten_solution.strip()
        else:
            rewritten_solution = ""

        return dspy.Prediction(
            rewritten_code=rewritten_solution,
        )


class BugInjector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.introduce_bug = dspy.Predict(IntroduceBug)

    def forward(self, task_prompt, gt_solution, bug_type, action_on_lines):
        """
        :param task_prompt: task
        :param gt_solution: solution
        :param bug_type: [type, definition, examples (list of tuples)]
        :param action_on_lines: [action, lines (list of tuples)]
        :return:
        """
        bug_type_str = f"{bug_type[0]}: {bug_type[1]}\nHere are a few examples in format (subtype, explanation, example):\n"
        for i, example in enumerate(bug_type[2]):
            bug_type_str += f"Example {i + 1}. {example[0]}: {example[1]}\n"
        bug_type_str += ("Please be very creative when introducing the bug! "
                         "Feel free to introduce a bug that does not fall into any of the above subtypes, but make sure that bug is challenging!"
                         "If so, output the subtype as \"Others\".")
        action_on_lines_str = f"{action_on_lines[0]}:\n"
        for line in action_on_lines[1]:
            action_on_lines_str += f"{line[0]}. {line[1]}\n"
        buggy_prediction = self.introduce_bug(
            task_prompt=task_prompt,
            correct_solution=gt_solution,
            bug_type=bug_type_str,
            action_on_lines=action_on_lines_str
        )
        if buggy_prediction.subtype:
            subtype = buggy_prediction.subtype
        else:
            subtype = ""

        if buggy_prediction.buggy_solution:
            match = CODE_BLOCK_REGEX.search(buggy_prediction.buggy_solution)
            match_simple = SIMPLE_CODE_BLOCK_REGEX.search(buggy_prediction.buggy_solution)
            if match:
                buggy_code = match.group(1).strip()
            elif match_simple:
                buggy_code = match.group(1).strip()
            else:
                buggy_code = buggy_prediction.buggy_solution.strip()
        else:
            buggy_code = ""

        return dspy.Prediction(
            subtype=subtype,
            buggy_code=buggy_code,
        )


class MultilineBugInjector(dspy.Module):
    def __init__(self, max_lines_per_block: int = MAX_MULTILINES):
        super().__init__()
        sig = make_multiline_bug_signature(MIN_MULTILINES, max_lines_per_block)
        self.introduce_bug = dspy.Predict(sig)
        self.max_lines = max_lines_per_block

    def forward(self, task_prompt, gt_solution, bug_type, action_on_lines):
        """
        action_on_lines: [action_str, list_of_contiguous_ranges]
        Each range is a list of (line_no, line_content) tuples forming a contiguous block.

        bug_type: [primary_category, definition, in_category_examples, cross_category_examples]
                  The 4th element (optional) is a list of (subtype, explanation) tuples drawn
                  from OTHER ODC categories to inspire cross-category blended bugs.
        """
        bug_type_str = f"{bug_type[0]}: {bug_type[1]}\nHere are a few examples in format (subtype, explanation, example):\n"
        for i, example in enumerate(bug_type[2]):
            bug_type_str += f"Example {i + 1}. {example[0]}: {example[1]}\n"

        # NOTE: [design thought] give the LLM cross-category inspiration so
        # the block can mix a Checking flip with an Assignment correction, etc.
        if len(bug_type) > 3 and bug_type[3]:
            bug_type_str += "\nYou may also blend in ideas from these other bug categories if it makes the multi-line bug more natural:\n"
            for i, (sub, expl) in enumerate(bug_type[3]):
                bug_type_str += f"Other {i + 1}. {sub}: {expl}\n"

        bug_type_str += (f"\nBe creative and think like a real human who made a mistake. "
                         f"The bug must span {MIN_MULTILINES}-{self.max_lines} contiguous lines where EVERY line is essential "
                         f"(reverting any single line to the GT must still leave the tests failing). "
                         f"If the subtype does not match any example, output the subtype as \"Others\".")

        # NOTE: [design thought] present contiguous
        # ranges (e.g., "Lines 10-13") instead of scattered individual lines.
        # This guides the LLM to produce contiguous edits that pass validation.
        action_on_lines_str = f"{action_on_lines[0]}:\n"
        for range_idx, line_range in enumerate(action_on_lines[1]):
            start = line_range[0][0]
            end = line_range[-1][0]
            action_on_lines_str += f"\nRange {range_idx + 1} (Lines {start}-{end}):\n"
            for line_no, line_content in line_range:
                action_on_lines_str += f"  {line_no}. {line_content}\n"

        buggy_prediction = self.introduce_bug(
            task_prompt=task_prompt,
            correct_solution=gt_solution,
            bug_type=bug_type_str,
            action_on_lines=action_on_lines_str
        )
        subtype = buggy_prediction.subtype if buggy_prediction.subtype else ""

        if buggy_prediction.buggy_solution:
            match = CODE_BLOCK_REGEX.search(buggy_prediction.buggy_solution)
            match_simple = SIMPLE_CODE_BLOCK_REGEX.search(buggy_prediction.buggy_solution)
            if match:
                buggy_code = match.group(1).strip()
            elif match_simple:
                buggy_code = match_simple.group(1).strip()
            else:
                buggy_code = buggy_prediction.buggy_solution.strip()
        else:
            buggy_code = ""

        return dspy.Prediction(
            subtype=subtype,
            buggy_code=buggy_code,
        )


class Debugger(dspy.Module):
    def __init__(self, model=None):
        super().__init__()
        self.minimal_debugger = dspy.Predict(MinimalDebug)
        self.minimal_feedback_debugger = dspy.Predict(MinimalFeedbackDebug)
        self.minimal_unit_feedback_debugger = dspy.Predict(MinimalUnitFeedbackDebug)
        self.free_debugger = dspy.Predict(FreeDebug)
        self.free_feedback_debugger = dspy.Predict(FreeFeedbackDebug)
        self.free_unit_feedback_debugger = dspy.Predict(FreeUnitFeedbackDebug)
        if model:
            self.external_model = ExternalModelWrapper(model)
        else:
            self.external_model = None

    def forward(self, task_prompt, buggy_code, test_cases=None, failures=None, mode="minimal"):
        """
        :param task_prompt: task
        :param buggy_code: buggy code
        :param test_cases: str of test cases
        :param failures: previously failed attempts and optionally error feedback
        :param mode: prompt variation on minimal debug or free debug
        :return:
        """
        if self.external_model:
            response = self.external_model(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
                unit_tests=test_cases,
                debug_mode=mode
            )
        elif mode == "minimal":
            response = self.minimal_debugger(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
            )
        elif mode == "minimal_with_feedback":
            assert failures is not None
            response = self.minimal_feedback_debugger(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
                failed_attempts=failures,
            )
        elif mode == "minimal_unit_with_feedback":
            assert failures is not None
            assert test_cases is not None
            response = self.minimal_unit_feedback_debugger(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
                failed_attempts=failures,
                unit_tests=test_cases,
            )
        elif mode == "free":
            response = self.free_debugger(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
            )
        elif mode == "free_with_feedback":
            assert failures is not None
            response = self.free_feedback_debugger(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
                failed_attempts=failures,
            )
        elif mode == "free_unit_with_feedback":
            assert failures is not None
            assert test_cases is not None
            response = self.free_unit_feedback_debugger(
                problem_description=task_prompt,
                buggy_solution=buggy_code,
                failed_attempts=failures,
                unit_tests=test_cases,
            )
        else:
            raise ValueError("Prompt mode not implemented")

        if response.corrected_solution:
            match = CODE_BLOCK_REGEX.search(response.corrected_solution)
            match_simple = SIMPLE_CODE_BLOCK_REGEX.search(response.corrected_solution)
            if match:
                output = match.group(1).strip()
            elif match_simple:
                output = match_simple.group(1).strip()
            else:
                output = response.corrected_solution.strip()
            if output.startswith("```"):
                output = output[3:].strip()
            if output.endswith("```"):
                output = output[:-3].strip()
        else:
            output = ""

        return dspy.Prediction(
            solution=output,
        )
