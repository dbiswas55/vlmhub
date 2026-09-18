"""Model/hosting registry — loads models.json and resolves a client config.

The bundled `models.json` ships with the package and covers the hostings and
models listed there. A project that needs more passes its own JSON file, which
is deep-merged over the bundled one: a key present in both takes the override's
value, and a dict is merged key by key. Lists are replaced wholesale, so an
override that names a hosting's "models" list replaces that hosting's models
rather than appending to them — add a new hosting key instead to extend.

An override file has the same flat shape as the bundled one: "active",
"defaults" and "hostings" at the top level.
"""

from __future__ import annotations

import json
from pathlib import Path


class Config:
    def __init__(self, models_path: str | Path | None = None):
        config = json.loads(self._bundled_text())
        if models_path is not None:
            override = json.loads(Path(models_path).read_text(encoding="utf-8"))
            config = self._deep_merge(config, override)
        self._config = config

    @staticmethod
    def _bundled_text() -> str:
        """The packaged models.json, read wherever the installed package lives."""
        from importlib.resources import files
        return files("vlmhub").joinpath("models.json").read_text(encoding="utf-8")

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            merged[key] = (Config._deep_merge(merged[key], value)
                           if isinstance(value, dict) and isinstance(merged.get(key), dict)
                           else value)
        return merged

    @staticmethod
    def _hosting_fields(hosting: dict) -> dict:
        """A hosting's own fields, minus its "models" list and "_"-prefixed comments."""
        return {k: v for k, v in hosting.items() if k != "models" and not k.startswith("_")}

    def _merged(self, hosting_name: str, hosting: dict, model: dict) -> dict:
        """One client dict: defaults < hosting fields < model fields."""
        defaults = self._config.get("defaults", {})
        return {**defaults, **self._hosting_fields(hosting), **model, "hosting": hosting_name}

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_current_client(self) -> dict:
        """The client named by "active" in the registry, merged with defaults."""
        active = self._config["active"]
        return self.get_client(active["hosting"], active["model"])

    def get_client(self, hosting_name: str, model_name: str) -> dict:
        """A specific client by hosting key and model name, merged with defaults."""
        hostings = self._config["hostings"]
        if hosting_name not in hostings or not isinstance(hostings[hosting_name], dict):
            available = sorted(k for k, v in hostings.items()
                               if isinstance(v, dict) and not k.startswith("_"))
            raise KeyError(f"No hosting '{hosting_name}' in models.json. "
                           f"Available: {', '.join(available)}")

        hosting = hostings[hosting_name]
        for model in hosting.get("models", []):
            if model["name"] == model_name:
                return self._merged(hosting_name, hosting, model)

        available = ", ".join(m["name"] for m in hosting.get("models", []))
        raise KeyError(f"No model '{model_name}' under hosting '{hosting_name}'. "
                       f"Available: {available}")

    def get_client_by_name(self, name: str) -> dict:
        """A client by 'hosting/model' name (e.g. 'ollama/gemma3-12b')."""
        if "/" not in name:
            raise KeyError(f"Invalid client name '{name}'. Use 'hosting/model' format "
                           f"(e.g. 'ollama/gemma3-12b').")
        hosting_name, model_name = name.split("/", 1)
        return self.get_client(hosting_name, model_name)

    def get_all_clients(self) -> list[dict]:
        """Every configured client across every hosting, merged with defaults."""
        clients = []
        for hosting_name, hosting in self._config["hostings"].items():
            if not isinstance(hosting, dict) or hosting_name.startswith("_"):
                continue
            for model in hosting.get("models", []):
                clients.append(self._merged(hosting_name, hosting, model))
        return clients
