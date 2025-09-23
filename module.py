import dspy
import re

MINIMAL_DEBUG_TEMPLATE = (
    "Analyze and debug the given Python implementation that contains errors \n"
    "Identify the bugs and fix ONLY the bugs in the code. Do not generate a new solution. You don't need to provide any explanation.\n\n"
    "You must preserve the original code logic exactly. Please Do NOT:\n"
    "- Change variable names\n"
    "- Modify a correct loop structure (e.g., for/while)\n"
    "- Remove or replace existing variables\n\n"
    "The input consists of two parts:\n"
    "PART 1: A problem description outlining the intended functionality.\n"
    "PART 2: A buggy implementation that needs to be fixed.\n"
    "Your response should include:\n"
    "- A self-contained, corrected Python implementation;\n"
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Buggy Code\n"
    "```python\n{buggy_code}\n```\n\n"
    "---\n"
    "Output format (follow *exactly*):\n"
    "```python\n[Corrected code here]\n```\n"
    "Corrected Code Output (use the format above):\n"
)

FREE_DEBUG_TEMPLATE = (
    "Analyze and debug the given Python implementation that contains errors \n"
    "Your response should include:\n"
    "- A self-contained, corrected Python implementation;\n"
    "\n"
    "---\n"
    "PART 1: Problem Description\n"
    "```text\n{task_prompt}\n```\n\n"
    "PART 2: Buggy Code\n"
    "```python\n{buggy_code}\n```\n\n"
    "---\n"
    "Output format (follow *exactly*):\n"
    "```python\n[Corrected code here]\n```\n"
    "Corrected Code Output (use the format above):\n"
)

CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DIFF_STR_PATTERN = re.compile(r"^(\d+): (.*) --> (.*)$")


class RewriteSolution(dspy.Signature):
    """Your task is to perform a deep analysis of code and rewrite it with structural and stylistic perturbations.
You will be given two parts:
PART 1: A problem description/requirement (the task that needs to be solved)
PART 2: A correct solution to the problem

First, carefully read both the problem description and the provided solution to understand what the code is supposed to do.
Then, your task is to perform a deep rewriting (perturbation) of the correct code implementation without changing its behavior or output.
The rewritten code should resemble what human will write and should NOT be hard for human to read.
Shallow redundancy edits such as a = 1 + 2 - 1 are not allowed.

Use the following rules and hints:
    - NEVER rename variables to very short names (e.g., sum_production -> sum_p), but you can give variables wrong names.
    - Remove all the comments and do not add any new comments.
    - Change loop syntax (e.g., for to while or vice versa, if applicable)
    - Convert recursion to iteration or vice versa (if functionally equivalent)
    - Invert control flow where logical structure remains the same (e.g., replace `if not ...` with an inverted block)
    - Merge or flatten adjacent or nested if blocks.
    - Keep the semantic of the code same but change syntax as much as you want.
    - Make the code slightly longer if possible.

Your response should ONLY contain the rewritten Python code, which can be directly executed.
Output format (follow *exactly*):
```python
[Your Rewritten Python Code Here]
```"""
    task_prompt = dspy.InputField(desc="The programming task description.")
    original_solution = dspy.InputField(desc="The original Python code solution.")
    rewritten_code = dspy.OutputField(desc="A functionally equivalent but different Python code solution.")


class IntroduceBug(dspy.Signature):
    """Your task is to perform a deep analysis of a code snippet and intentionally introduce one bug.
You will be given two components:
PART 1: A task description outlining the intended functionality
PART 2: A solution to the task.
First, carefully read and understand both the task description and the provided solution.
You will be asked to introduce a certain type of bug to the code: adding a line, deleting a line or modify a line.
Then, modify the solution by injecting realistic programming errors to simulate human mistakes.

Instructions for modifying the code:
- Delete all comments from the original code and Do NOT add any new comments to the modified code.
- DO NOT change any other variable names.
- DO NOT introduce easy bugs such as referencing variable names before declaration, typing errors.
- Please ONLY choose and modify one line to induce a HARD bug to the task.

Output format:
- A single code block with the final, modified version of the buggy code with NO comments.
You need to check there is NO COMMENT inside your generation for the final step.
Output format (follow *exactly*):
```python
[Your Buggy Code Here]
```"""
    task_prompt = dspy.InputField(desc="The programming task description for context.")
    correct_solution = dspy.InputField(desc="A correct Python code solution.")
    bug_type = dspy.InputField(desc="Type of bug to add.")
    buggy_solution = dspy.OutputField(desc="The same Python code with a single-line bug introduced.")


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteSolution)

    def forward(self, task_prompt, gt_solution):
        prediction = self.rewrite(task_prompt=task_prompt, original_solution=gt_solution)
        rewritten_code = prediction.rewritten_code

        return dspy.Prediction(
            rewritten_code=rewritten_code,
        )


class BugInjector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.introduce_bug = dspy.Predict(IntroduceBug)

    def forward(self, task_prompt, gt_solution, bug_type):
        buggy_prediction = self.introduce_bug(
            task_prompt=task_prompt,
            correct_solution=gt_solution,
            bug_type=bug_type,

        )
        buggy_code = buggy_prediction.buggy_solution

        return dspy.Prediction(
            buggy_code=buggy_code
        )
