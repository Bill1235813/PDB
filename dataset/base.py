import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from utils import get_indentation
from module import SIMPLE_CODE_BLOCK_REGEX


# NOTE: We use an ABC rather than duck typing so that forgetting to implement
# a method raises an error at import time, not deep into a long evaluation run.
class DatasetHandler(ABC):
    """
    Abstract base class for dataset-specific operations in PDB.

    Each supported dataset must subclass this and implement the abstract methods below.
    These methods encapsulate dataset-dependent logic that was previously dispatched
    via string comparisons in utils.py.

    Subclasses must implement:
        - preprocess(raw_data)
        - verify_unit_test(verify_file, ...)
        - build_verify_unit_test(log_file_prefix, results, ...)
        - save_formatted_gt(log_file_prefix, data)

    The mark_editable_lines method has a concrete default implementation shared by
    BigCodeBench and LiveCodeBench. Override it if your dataset has different rules.
    """

    # NOTE: [pedagogical] These keyword lists define which lines are "structural" and should
    # not be modified (NO_CHANGE_KEYWORDS) or deleted (NO_DELETE_KEYWORDS) during bug injection.
    # Modifying structural lines (e.g., function signatures, imports) would create unrealistic
    # bugs that don't reflect real-world debugging scenarios.
    NO_CHANGE_KEYWORDS = ["try", "except", "finally", "import", "def", "class", "async"]
    NO_DELETE_KEYWORDS = ["if", "else", "elif", "for", "while", "with"]

    # Subdirectory (relative to the subclass's file) that holds the dataset's
    # self-contained uv install tree. Subclasses can override if needed.
    install_subdir: ClassVar[str] = "install"

    # Whether parse_diff_to_blocks merges Add(N) into an adjacent Delete/Modify(N-1)
    # block (see utils.set_block_merge_mode). Off for the original PDB datasets.
    merge_adjacent_add_blocks: ClassVar[bool] = False

    # Optional hooks for the coverage gate / test augmentation (src/test_adequacy.py).
    # Handlers that support them implement:
    #   measure_line_coverage(tasks, extra_tests=None, extra_setup="") -> {task_id: {"hit_lines": [...]}}
    #   run_tests_in_container(repo, files, test_ids, extra_setup="") -> {test_id: status}
    #   augmented_test_path(task, index) -> str ; module_name(task) -> str
    # and run a task's `augmented_test_ids` during verify_unit_test.

    @property
    def install_dir(self) -> Path:
        """Absolute path to this dataset's uv install directory."""
        return Path(inspect.getfile(type(self))).parent / self.install_subdir

    @property
    def venv_python(self) -> Path:
        """Python interpreter inside the dataset's uv-managed .venv."""
        return self.install_dir / ".venv" / "bin" / "python"

    def venv_cmd(self, module: str, *args: str) -> list[str]:
        """
        Build a subprocess command that runs ``python -m <module> <args...>``
        inside this dataset's uv virtualenv. The subprocess is fully isolated:
        the parent shell's environment is untouched, and the venv lifetime
        matches the subprocess lifetime — no activate/deactivate needed.
        """
        if not self.venv_python.exists():
            raise RuntimeError(
                f"uv venv missing at {self.venv_python}. "
                f"Run `cd {self.install_dir} && uv sync` "
                f"(see {self.install_dir.parent}/README.md for full setup)."
            )
        return [str(self.venv_python), "-m", module, *args]

    def mark_editable_lines(self, data):
        """
        Compute which lines in each code snippet are editable and deletable.

        Mutates each dict in data in-place, adding:
          - "frozen_lines": int, number of lines from the starter code that must not change
          - "gt_length": int, total number of lines in the ground truth solution
          - "editable_lines": list of (line_number, line_content) tuples
          - "deletable_lines": list of (line_number, line_content) tuples

        NOTE: [pedagogical] "Editable" means the line can be modified by the bug injector.
        "Deletable" is a stricter subset — lines that can be removed without breaking syntax
        (e.g., not the only statement in a block, not a control flow keyword).
        """
        if len(data) == 0:
            return

        assert "task_prompt" in data[0]
        for d in data:
            # NOTE: The frozen_lines count comes from the starter code block
            # in the task prompt. Lines within the starter code are "frozen" — they set up the
            # function signature and imports that the test harness depends on.
            code_matches = SIMPLE_CODE_BLOCK_REGEX.findall(d["task_prompt"])
            if code_matches:
                d["frozen_lines"] = len(code_matches[-1].strip().splitlines())
            else:
                d["frozen_lines"] = 0

            code_lines = d["gt_solution"].splitlines()
            code_length = len(code_lines)
            d["gt_length"] = code_length
            d["editable_lines"] = []
            d["deletable_lines"] = []

            for i, line in enumerate(code_lines):
                if i >= d["frozen_lines"] and line.strip() != "":
                    stripped_line = line.strip()
                    editable = True
                    deletable = True

                    for keyword in self.NO_CHANGE_KEYWORDS:
                        if keyword in stripped_line:
                            editable = False
                            break
                    for keyword in self.NO_DELETE_KEYWORDS:
                        if keyword in stripped_line:
                            deletable = False
                            break

                    if editable and deletable:
                        if stripped_line.endswith(":"):
                            deletable = False
                        elif i > 0 and code_lines[i - 1].strip().endswith(":"):
                            # NOTE: [edge case callout] Cannot delete the first (and only) statement
                            # in a block — that would leave a colon with an empty body, which is a
                            # syntax error.
                            if i == len(code_lines) - 1:
                                deletable = False
                            else:
                                prev_indent = get_indentation(code_lines[i - 1])
                                next_indent = get_indentation(code_lines[i + 1])
                                if prev_indent == next_indent:
                                    deletable = False
                        elif (0 < i < len(code_lines) - 1 and not code_lines[i - 1].strip()
                              and not code_lines[i + 1].strip()):
                            # NOTE: [edge case callout] Don't delete a line surrounded by blank
                            # lines — it would create a triple blank gap that may confuse parsers.
                            deletable = False

                    if editable:
                        d["editable_lines"].append((i + 1, line))
                    if editable and deletable:
                        d["deletable_lines"].append((i + 1, line))

    @abstractmethod
    def preprocess(self, raw_data):
        """
        Transform raw dataset files into the standardized PDB format.

        Returns:
            list[dict]: Each dict has at least "task_id", "gt_solution", "task_prompt".
        """
        ...

    @abstractmethod
    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=20, timeout=1800, n_workers=None):
        """
        Run unit tests using this dataset's evaluation harness.

        n_workers is a parallelism hint; handlers whose harness manages its own
        parallelism may ignore it.

        Returns:
            (fail_ids, correct_ids, fail_feedback):
            - fail_ids: list of task_id strings that failed
            - correct_ids: list of task_id strings that passed
            - fail_feedback: dataset-specific failure details (str or list)
        """
        ...

    @abstractmethod
    def build_verify_unit_test(self, log_file_prefix, results, sol_field="solution"):
        """
        Build the verification file(s) that verify_unit_test() will consume.

        Returns:
            str or None: Path to the verification file, or None if nothing to verify.
        """
        ...

    @abstractmethod
    def save_formatted_gt(self, log_file_prefix, data):
        """
        Save ground truth data in the format this dataset's evaluator expects.

        Returns:
            str or None: Path to the ground truth file, or None if not needed.
        """
        ...
