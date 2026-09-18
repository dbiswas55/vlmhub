"""Local model store management — list, download, and delete models cached
by HuggingFace (used by MLX-VLM, vLLM, and Transformers) and by Ollama.
The two stores are independent (different directories, different formats);
this file just gives them a matching set of functions.

Full backend setup guide (venv, Ollama, MLX-VLM, vLLM): see
../backends/README.md. Run this file directly to inspect, download into, and
prune both stores:
    python src/vlmhub/utils/local_models.py

`ollama` is an optional import, needed only for the Ollama functions.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download, scan_cache_dir

# Load HF_TOKEN and HF_HOME from .env before any huggingface_hub calls
load_dotenv()


# ── HuggingFace Cache ─────────────────────────────────────────────────────────

def list_hf_cache_models():
    """List all models in the HF cache with their sizes."""
    cache_info = scan_cache_dir()
    repos = sorted(cache_info.repos, key=lambda r: r.size_on_disk, reverse=True)

    if not repos:
        print("No models found in HF cache.")
        return

    total = sum(r.size_on_disk for r in repos)
    print("\n=== HuggingFace Cache ===")
    print(f"{'#':<4} {'Model':<55} {'Size':>10}  Revisions")
    print("-" * 85)
    for i, repo in enumerate(repos, 1):
        size_gb = repo.size_on_disk / 1024 ** 3
        revisions = len(repo.revisions)
        print(f"{i:<4} {repo.repo_id:<55} {size_gb:>8.2f} GB  {revisions}")
    print("-" * 85)
    print(f"{'Total:':<60} {total / 1024 ** 3:>8.2f} GB\n")
    return repos


def hf_login():
    """Interactive HF login — only needed for gated models not covered by HF_TOKEN in .env."""
    login()


def download_hf_model(model_ids: str | list[str]):
    """Download one or more model repos into the local HF cache (HF_HOME)."""
    if isinstance(model_ids, str):
        model_ids = [model_ids]
    for model_id in model_ids:
        print(f"Downloading {model_id} ...")
        snapshot_download(repo_id=model_id, token=os.getenv("HF_TOKEN"))
        print(f"Done: {model_id}")


def delete_hf_cache_model(model_id: str):
    """Delete a specific model from the HF cache by repo id."""
    cache_info = scan_cache_dir()
    matches = [r for r in cache_info.repos if r.repo_id == model_id]

    if not matches:
        print(f"Model '{model_id}' not found in HF cache.")
        return

    repo = matches[0]
    size_gb = repo.size_on_disk / 1024 ** 3
    confirm = input(f"Delete '{model_id}' ({size_gb:.2f} GB)? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Mark all revisions for deletion and commit
    delete_strategy = cache_info.delete_revisions(
        *[rev.commit_hash for rev in repo.revisions]
    )
    print(f"Freeing {delete_strategy.expected_freed_size_str} ...")
    delete_strategy.execute()
    print(f"Deleted '{model_id}' from HF cache.")


def delete_hf_cache_model_interactive():
    """Show cached HF models and prompt the user to pick one to delete, repeating until cancelled."""
    while True:
        repos = list_hf_cache_models()
        if not repos:
            break
        choice = input("Enter the # of the model to delete (or 0 to cancel): ").strip()
        if not choice.isdigit() or int(choice) == 0:
            print("Cancelled.")
            break
        if 1 <= int(choice) <= len(repos):
            delete_hf_cache_model(repos[int(choice) - 1].repo_id)
        else:
            print(f"Invalid choice, pick a number between 1 and {len(repos)}.")


# ── Ollama Store ──────────────────────────────────────────────────────────────

def _ollama():
    """Lazy-import the optional `ollama` package; None (with a message) if missing."""
    try:
        import ollama
        return ollama
    except ImportError:
        print("Skipping Ollama: pip install ollama")
        return None


def list_ollama_cache_models():
    """List all models in Ollama's local store with their sizes."""
    ollama = _ollama()
    if not ollama:
        return
    models = ollama.list().models
    if not models:
        print("No models found in Ollama.")
        return

    models = sorted(models, key=lambda m: m.size, reverse=True)
    total = sum(m.size for m in models)
    print("\n=== Ollama Local Store ===")
    print(f"{'#':<4} {'Model':<30} {'Size':>10}  {'Params':<8} Quantization")
    print("-" * 75)
    for i, m in enumerate(models, 1):
        size_gb = m.size / 1024 ** 3
        print(f"{i:<4} {m.model:<30} {size_gb:>8.2f} GB  {m.details.parameter_size:<8} {m.details.quantization_level}")
    print("-" * 75)
    print(f"{'Total:':<45} {total / 1024 ** 3:>8.2f} GB\n")
    return models


def download_ollama_model(model_ids: str | list[str]):
    """Download one or more models into Ollama's local store, printing progress."""
    ollama = _ollama()
    if not ollama:
        return
    if isinstance(model_ids, str):
        model_ids = [model_ids]
    for model_id in model_ids:
        print(f"Downloading {model_id} ...")
        for progress in ollama.pull(model_id, stream=True):
            pct = f" {progress.completed / progress.total:.0%}" if progress.total else ""
            print(f"\r{progress.status}{pct}   ", end="", flush=True)
        print(f"\nDone: {model_id}")


def delete_ollama_cache_model(model_id: str):
    """Delete a specific model from Ollama's local store by name."""
    ollama = _ollama()
    if not ollama:
        return
    if model_id not in {m.model for m in ollama.list().models}:
        print(f"Model '{model_id}' not found in Ollama.")
        return

    confirm = input(f"Delete '{model_id}' from Ollama? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    ollama.delete(model_id)
    print(f"Deleted '{model_id}' from Ollama.")


def delete_ollama_cache_model_interactive():
    """Show local Ollama models and prompt the user to pick one to delete, repeating until cancelled."""
    while True:
        models = list_ollama_cache_models()
        if not models:
            break
        choice = input("Enter the # of the model to delete (or 0 to cancel): ").strip()
        if not choice.isdigit() or int(choice) == 0:
            print("Cancelled.")
            break
        if 1 <= int(choice) <= len(models):
            delete_ollama_cache_model(models[int(choice) - 1].model)
        else:
            print(f"Invalid choice, pick a number between 1 and {len(models)}.")


if __name__ == "__main__":
    # ── List: see what's already there ──────────────────────────────────────
    list_hf_cache_models()
    list_ollama_cache_models()

    # ── Download: uncomment or add the models you want, then run this file ──
    # Requires: HF_TOKEN in .env, or hf_login() for interactive auth.
    # hf_login()
    hf_models = [
        # "google/gemma-3-12b-it",
        # "mlx-community/gemma-3-12b-it-qat-4bit",
        # "mlx-community/gemma-3-4b-it-qat-4bit",
        # "Qwen/Qwen3-VL-4B-Instruct",
    ]
    # download_hf_model(hf_models)

    ollama_models = [
        # "gemma3:12b",
        # "qwen3-vl:8b",
    ]
    # download_ollama_model(ollama_models)

    # ── Delete: interactive pickers ──────────────────────────────────────────
    # delete_hf_cache_model_interactive()
    # delete_ollama_cache_model_interactive()
