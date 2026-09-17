"""Backend factory — instantiates the correct backend from a config client dict."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from .backends import BaseBackend, GeminiBackend, OpenAIBackend, TransformersBackend

load_dotenv()



def get_backend_from_config(client: dict) -> BaseBackend:
    """Instantiate the correct backend from a config client dict."""
    name    = client["name"]
    backend = client["backend"]
    model_id = client["model_id"]

    api_key_env = client.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None

    if backend == "gemini":
        vertexai_project_env = client.get("vertexai_project_env")
        vertexai_location_env = client.get("vertexai_location_env")
        vertexai_project = os.getenv(vertexai_project_env) if vertexai_project_env else None
        vertexai_location = os.getenv(vertexai_location_env) if vertexai_location_env else None

        return GeminiBackend(
            name=name,
            model_id=model_id,
            api_key=api_key,
            thinking_budget=client.get("thinking_budget"),
            vertexai_project=vertexai_project,
            vertexai_location=vertexai_location,
        )

    if backend == "openai":
        base_url = client.get("base_url", "https://api.openai.com/v1")
        # Use "no-key" for local servers that don't require authentication
        return OpenAIBackend(name=name, model_id=model_id, base_url=base_url, api_key=api_key or "no-key")

    if backend == "transformers":
        _hf_token = os.getenv("HF_TOKEN")
        _hf_home = os.getenv("HF_HOME") or None

        return TransformersBackend(
            name=name,
            hf_model_id=model_id,
            hf_token=_hf_token,
            hf_cache=_hf_home,
            quantization_level=client.get("quantization_level"),
            fallback_dtype=client.get("fallback_dtype"),
            model_class=client.get("model_class"),
            processor_kwargs=client.get("processor_kwargs"),
        )

    raise ValueError(
        f"Unknown backend '{backend}' for client '{name}'. "
        "Expected one of: gemini, openai, transformers."
    )
