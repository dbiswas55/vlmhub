"""Experiment configuration — loads a JSON config file and provides
structured accessors for models, datasets, task params, and prompts."""

from __future__ import annotations

import json
from pathlib import Path


class Config:
    def __init__(self, config_path: str = "configs/experiment.json"):
        self.config_path = Path(config_path).resolve()
        self.project_dir   = self.config_path.parent.parent
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    # ── Generic accessor ──────────────────────────────────────────────────────

    def get(self, *keys):
        """Traverse nested keys and return the value.

        Example:
            cfg.get("datasets", "demo_images")       # → {"name": "demo_images", ...}
            cfg.get("models", "active", "hosting")   # → "ollama"
        """
        result = self.config
        for key in keys:
            result = result[key]
        return result

    def resolve(self, path: str) -> Path:
        """A config path resolved against project_dir, "~" expanded first so an
        already-absolute value is kept as-is rather than nested under it."""
        return (self.project_dir / Path(path).expanduser()).resolve()

    # ── Model / client ────────────────────────────────────────────────────────

    def get_current_client(self) -> dict:
        """Return the active client (from 'active.hosting' + 'active.model') merged with defaults."""
        active = self.config["models"]["active"]
        return self.get_client(active["hosting"], active["model"])

    @staticmethod
    def _hosting_fields(hosting: dict) -> dict:
        """A hosting's own fields, minus its "models" list and "_"-prefixed comments."""
        return {k: v for k, v in hosting.items() if k != "models" and not k.startswith("_")}

    def get_client(self, hosting_name: str, model_name: str) -> dict:
        """Return a specific client by hosting key and model name, merged with defaults."""
        hostings = self.config["models"]["hostings"]
        if hosting_name not in hostings:
            raise KeyError(f"No hosting '{hosting_name}' found in config.")

        hosting = hostings[hosting_name]
        if not isinstance(hosting, dict):
            raise KeyError(f"'{hosting_name}' is a comment entry, not a hosting.")

        defaults = self.config["models"].get("defaults", {})
        for model in hosting.get("models", []):
            if model["name"] == model_name:
                return {**defaults, **self._hosting_fields(hosting), **model, "hosting": hosting_name}
        raise KeyError(f"No model '{model_name}' found under hosting '{hosting_name}'.")

    def get_all_clients(self) -> list[dict]:
        """Return all configured clients across all hostings, merged with defaults."""
        defaults = self.config["models"].get("defaults", {})

        clients = []
        for hosting_name, hosting in self.config["models"]["hostings"].items():
            if not isinstance(hosting, dict) or hosting_name.startswith("_"):
                continue
            for model in hosting.get("models", []):
                clients.append({**defaults, **self._hosting_fields(hosting), **model, "hosting": hosting_name})
        return clients

    def get_client_by_name(self, name: str) -> dict:
        """Return a client by 'hosting/model' name (e.g. 'ollama/gemma3-12b')."""
        if "/" not in name:
            raise KeyError(f"Invalid client name '{name}'. Use 'hosting/model' format (e.g. 'ollama/gemma3-12b').")
        hosting_name, model_name = name.split("/", 1)
        return self.get_client(hosting_name, model_name)

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def get_task_config(self, task: str) -> dict:
        """Return one "tasks" entry — a script's general parameters.

        These are defaults: a script overrides them with whatever was actually
        passed on its command line, which keeps the parameter out of the argument
        parser's defaults. Usually reached through get_task_params() below, which
        adds the dataset the task names. "_"-prefixed keys are comments and are
        dropped.

        Example:
            cfg.get_task_config("demo_summary")
            # → {"dataset": "demo_images", "reprocess": False}
        """
        tasks = self.config.get("tasks", {})
        if task not in tasks:
            raise KeyError(f"No task '{task}' in {self.config_path}. "
                           f"Available: {sorted(k for k in tasks if not k.startswith('_'))}")
        return {k: v for k, v in tasks[task].items() if not k.startswith("_")}

    def get_task_params(self, task: str) -> dict:
        """One task's entry flattened with the "datasets" entry it names.

        A task always identifies its dataset by a "dataset" key, so the two
        belong to one run and come back as a single flat dict. Models and
        workflows are chosen per run rather than fixed by the task, so they stay
        separate -- see get_client* and get_workflow_steps. "_"-prefixed keys are
        comments and are dropped from both halves.

        Example:
            cfg.get_task_params("demo_summary")
            # → {"dataset": "demo_images", "reprocess": False,   # the task
            #    "input_dir": "input/images", ...}               # the dataset
        """
        params = self.get_task_config(task)
        if "dataset" not in params:
            raise KeyError(f"Task '{task}' has no \"dataset\" key naming which dataset "
                           f"it runs on, in {self.config_path}.")
        params.update({k: v for k, v in self.get_dataset_config(params["dataset"]).items()
                       if not k.startswith("_")})
        return params

    # ── Datasets ──────────────────────────────────────────────────────────────

    def get_dataset_config(self, dataset_name: str) -> dict:
        """Return the config block for a named dataset.

        Example:
            cfg.get_dataset_config("demo_images")
            # → {"name": "demo_images", "input_dir": "input/images", "output_dir": "output/demo_images"}
        """
        return self.config["datasets"][dataset_name]

    # ── Prompts ───────────────────────────────────────────────────────────────

    def get_workflow_steps(self, workflow: str) -> list:
        """Return all steps for a workflow.

        Example:
            steps = cfg.get_workflow_steps("multisteps_summary")  # → list of step dicts
        """
        return self.config["prompts"]["workflows"][workflow]["steps"]

    def get_step(self, workflow: str, step: int) -> dict:
        """Return a single step dict (contains 'system' and 'user' keys).

        step is 1-based (step=1 is the first step, step=2 is the second, etc.).

        Example:
            step = cfg.get_step("multisteps_summary", 1)
            # → {"system": "", "user": "summary/v2_step1.txt"}
        """
        return self.config["prompts"]["workflows"][workflow]["steps"][step - 1]

    def get_step_text(self, workflow: str, step: int, prompt_type: str, version: str | None = None) -> str:
        """Return the resolved text for a prompt field in a step.

        step is 1-based (step=1 is the first step, step=2 is the second, etc.).

        - If the value ends with '.txt', it is treated as a file path relative
          to prompt_root and its contents are returned, filling in any
          "{version}" placeholder first.
        - Otherwise the value is returned as a literal string (e.g. an inline
          system prompt or an empty string).

        Example:
            user_text   = cfg.get_step_text("multisteps_summary", 1, "user")
            system_text = cfg.get_step_text("multisteps_summary", 1, "system")  # → ""
        """
        value = self.get_step(workflow, step)[prompt_type]
        if value and value.endswith(".txt"):
            filename = value.format(version=version) if version is not None else value
            root = self.config["prompts"]["prompt_root"]
            path = self.project_dir / root / filename
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return value
