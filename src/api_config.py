"""
Default model-to-API-key-file mapping.

Maps model name prefixes to their corresponding key files under the keys/ directory.
This allows scripts to resolve API keys automatically without requiring
--model_api_file every time.
"""
import os

# NOTE: [design thought] We match by prefix so that model variants
# (e.g. "openai/gpt-5.1-codex", "openai/gpt-4o") all resolve to the same key.
# More specific prefixes should come before general ones.
MODEL_KEY_MAP = {
    "together_ai/": "together_key.txt",
    "openai/": "openai_key.txt",
    "anthropic/": "anthropic_key.txt",
    "gemini/": "google_key.txt",
    "deepseek/": "deepseek_key.txt",
    "xai/": "xai_key.txt",
    "openrouter/": "openrouter_key.txt",
    "vertex_ai/": None,  # ADC; returns None so caller omits api_key
}


def resolve_api_key(model_name, model_api_file=None, keys_dir="keys"):
    """
    Resolve the API key for a given model name.

    If model_api_file is provided, use that directly. Otherwise, look up the
    model name prefix in MODEL_KEY_MAP to find the default key file.

    Returns the API key string, or None for providers that authenticate with
    ambient credentials (e.g. vertex_ai/ via Application Default Credentials).
    """
    if model_api_file:
        return open(os.path.join(keys_dir, model_api_file), "r").read().strip()

    for prefix, key_file in MODEL_KEY_MAP.items():
        if model_name.startswith(prefix):
            if key_file is None:
                return None
            return open(os.path.join(keys_dir, key_file), "r").read().strip()

    raise ValueError(
        f"No API key mapping found for model '{model_name}'. "
        f"Either pass --model_api_file or add a mapping in api_config.py."
    )
