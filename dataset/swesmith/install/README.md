# SWE-smith / SWE-bench (vendored)

The SWE-smith handler imports these two packages directly; the handler adds both
directories to `sys.path`, so they are not pip-installed. Their runtime
dependencies come from the `swesmith` extra of the top-level `pyproject.toml`.

| directory | package | version |
|---|---|---|
| `SWE-smith/swesmith` | [SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith) | 0.0.9 |
| `SWE-bench/swebench` | [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | 4.1.0 |

Only the Python packages, `LICENSE` and `pyproject.toml` of each project are
kept (tests, docs and assets are dropped). `swesmith.harness.valid` imports
`swebench.harness.*`, so both are needed.

To use a different version, replace a package directory with the one from a
fresh clone, e.g.:

```bash
git clone --depth 1 https://github.com/SWE-bench/SWE-smith.git /tmp/SWE-smith
rm -rf SWE-smith/swesmith && cp -r /tmp/SWE-smith/swesmith SWE-smith/
```
