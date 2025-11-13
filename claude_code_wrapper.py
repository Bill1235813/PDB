"""
Minimal Claude Code wrapper acts as drop-in replacement for dspy.LM.
"""
import subprocess


class ClaudeCodeGenerator:

    def __init__(self, temperature=0.7, max_tokens=4000, timeout=300):
        """
        Only keep for compatibility, actually claude code doesn't use it
        """
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        # prerequisite: Verify claude CLI is available
        try:
            subprocess.run(['claude', '--version'], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Claude Code CLI not found. Install from: https://claude.ai/download")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code CLI verification timed out")

    def __call__(self, prompt):
        try:
            result = subprocess.run(
                ['claude', '-p', '--no-memory', prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False 
            )

            if result.returncode != 0:
                print(f"Claude Code exited with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                # Return stdout if available, otherwise empty string
                return [result.stdout if result.stdout else ""]

            raw_response = result.stdout
            return [raw_response]

        except subprocess.TimeoutExpired:
            print(f"Claude Code timed out after {self.timeout}s")
            return [""]
        except Exception as e:
            print(f"Error invoking Claude Code: {e}")
            return [""]