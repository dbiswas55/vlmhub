"""The Model class — a client's config plus the backend that serves it."""

from __future__ import annotations

from pathlib import Path

from .utils.config import Config
from .backends import get_backend_from_config
from .backends.request import ContentBlock, InferenceRequest


class Model:
    """The client this run calls: its config, plus the backend that serves it.

        model = Model()                          # models.json's "active" client
        model = Model("ollama/gemma3-4b")        # a specific one from the bundled registry
        model = Model("custom/x", models_path="my_models.json")  # extend the registry
        reply = model.generate([TextBlock("Hello")])   # {"text": str, "logs": list[str]}
    """

    def __init__(self, client_name: str = "", models_path: str | Path | None = None):
        registry = Config(models_path)
        self.config = registry.get_client_by_name(client_name) if client_name else registry.get_current_client()
        self.backend = get_backend_from_config(self.config)

    @property
    def name(self) -> str:
        """This client as 'hosting/model' — the name it is selected by."""
        return f"{self.config['hosting']}/{self.config['name']}"

    @property
    def model_id(self) -> str:
        """The provider's own id for this model."""
        return self.config["model_id"]

    @property
    def short_name(self) -> str:
        """The model's alias within its hosting, without the hosting prefix."""
        return self.config["name"]

    def generate(self, content: list[ContentBlock], system_prompt: str = "", **overrides) -> dict:
        """Run one request; overrides may set max_new_tokens/temperature/top_p for this call only."""
        request = InferenceRequest(
            content=content,
            system_prompt=system_prompt,
            max_new_tokens=overrides.get("max_new_tokens", self.config["max_tokens"]),
            temperature=overrides.get("temperature", self.config["temperature"]),
            top_p=overrides.get("top_p", self.config["top_p"]),
        )
        return self.backend.run(request)

    def report(self) -> None:
        print(f"Client   : {self.name}  ->  {self.model_id}")
