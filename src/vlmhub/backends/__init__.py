"""Backend factory — instantiates the correct backend from a config client dict."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from .backends import BaseBackend, LiteLLMBackend, TransformersBackend

load_dotenv()


def get_backend_from_config(client: dict) -> BaseBackend:
    """Instantiate the correct backend from a config client dict."""
    name = client["name"]
    backend = client["backend"]

    if backend == "litellm":
        # LiteLLM routes by "<provider>/<model_id>"; no prefix means model_id is passed through as-is.
        litellm_prefix = client.get("litellm_prefix")
        model_id = client["model_id"]
        litellm_model = f"{litellm_prefix}/{model_id}" if litellm_prefix else model_id

        # A real secret is named via "api_key_env". Failing that, a local
        # OpenAI-compatible server (Ollama, MLX-VLM) wants *a* token but never
        # checks it, so it gets a harmless placeholder automatically.
        api_key_env = client.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)  # real secret from .env
        elif litellm_prefix == "openai" and client.get("api_base"):
            api_key = "not-needed"  # local OpenAI-compatible server, no real key needed
        else:
            api_key = None  # no key configured and none needed (e.g. vllm's own server)

        vertex_project = os.getenv(client["vertex_project_env"]) if client.get("vertex_project_env") else None
        vertex_location = os.getenv(client["vertex_location_env"]) if client.get("vertex_location_env") else None

        return LiteLLMBackend(
            name=name,
            litellm_model=litellm_model,
            api_base=client.get("api_base"),
            api_key=api_key,
            thinking_budget=client.get("thinking_budget"),
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )

    if backend == "transformers":
        return TransformersBackend(
            name=name,
            hf_model_id=client["model_id"],
            hf_token=os.getenv("HF_TOKEN"),
            hf_cache=os.getenv("HF_HOME") or None,
            quantization_level=client.get("quantization_level"),
            fallback_dtype=client.get("fallback_dtype"),
            model_class=client.get("model_class"),
            processor_kwargs=client.get("processor_kwargs"),
        )

    raise ValueError(
        f"Unknown backend '{backend}' for client '{name}'. "
        "Expected 'litellm' or 'transformers'."
    )
