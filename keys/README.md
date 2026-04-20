# API keys

PDB resolves provider keys by **model-name prefix**. Each file holds a single line, the raw API key for that provider.

The repository ships **placeholder files** (`YOUR_KEY_HERE`) for the providers used in the paper so you only have to overwrite them — no need to create the files from scratch:

| prefix in `--model_name` | key file |
|---|---|
| `openai/` | `keys/openai_key.txt` |
| `anthropic/` | `keys/anthropic_key.txt` |
| `gemini/` | `keys/google_key.txt` |
| `deepseek/` | `keys/deepseek_key.txt` |
| `xai/` | `keys/xai_key.txt` |
| `together_ai/` | `keys/together_key.txt` |

Full mapping in [src/api_config.py](../src/api_config.py). To point at a different file for a single run, pass `--model_api_file <filename>` (relative to this directory).

`.gitignore` covers the whole `keys/` directory. The placeholder files above and this README are the only tracked files; once you replace a placeholder with a real key, it will no longer show up in `git status`. **Never commit a real key** — if it ever gets staged, rotate the key and restore the placeholder.

For local / self-hosted model endpoints (vLLM, Ollama, LM Studio, …) see [scripts/README.md](../scripts/README.md#local--self-hosted-model-evaluation).
