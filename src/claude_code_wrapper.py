"""
Minimal Claude Code CLI wrapper that acts as a drop-in replacement for dspy.LM.

Used by `bug_correct.py` when `--use_claude_code` is passed: instead of a
dspy.LM-backed debugger, we shell out to the `claude` CLI with the prompt and
return its stdout. This lets PDB evaluate Claude Code's end-to-end agentic
debugging behavior alongside single-call LMs.
"""
import subprocess


class ClaudeCodeGenerator:

    def __init__(self, temperature=1.0, max_tokens=32000, timeout=300):
        """
        NOTE: [design thought] temperature and max_tokens are accepted only to
        match the dspy.LM constructor signature; the Claude Code CLI manages
        sampling on its own and ignores both. We keep them so callers can
        instantiate this class interchangeably with dspy.LM.
        """
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Preflight: fail fast if the CLI is missing so the caller sees a clear
        # error before any debug task runs.
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Claude Code CLI not found. Install from: https://claude.ai/download")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code CLI verification timed out")

    def __call__(self, prompt):
        # NOTE: [design thought] dspy.LM returns a list of completions, so we
        # wrap the single Claude Code stdout in a one-element list to keep the
        # call site identical.
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"Claude Code timed out after {self.timeout}s")
            return [""]

        if result.returncode != 0:
            # NOTE: [edge case callout] Non-zero exit with a non-empty stdout
            # still happens (e.g. partial output before a crash); return
            # whatever the CLI produced so downstream parsing can salvage it.
            print(f"Claude Code exited with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return [result.stdout or ""]

        return [result.stdout]
