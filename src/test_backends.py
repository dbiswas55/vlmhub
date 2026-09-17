"""
test_backends.py
================
Run a single multimodal inference request against a configured backend.

The smallest check that a client/backend is wired up correctly: resolve one
client from configs/experiment.json, build one image + text request, send it,
and print the response. For multi-step prompt workflows over a dataset, see
test_workflows.py.

Configuration
-------------
  Top-level constants (edit directly in this file):
    CLIENT_NAME   — client as "hosting/model", or "" to use config's active client
    IMAGE_PATHS   — image files to send alongside the prompt
    SYSTEM_PROMPT — optional system prompt ("" = none)
    USER_PROMPT   — the instruction sent with the images
    DEBUG         — if True, print the resolved client and request before running

Usage
-----
    python src/test_backends.py
"""

from time import time
from pathlib import Path

from utils.config import Config
from backends import get_backend_from_config
from backends.request import TextBlock, ImageBlock, InferenceRequest


# ── Top-level constants — edit these to control the request ─────────────────
# Available hostings : gemini, gemini_vtx, gemini_oai, openai, ollama, mlx_vlm, vllm, transformers
# Available models   : gemma3-4b, gemma3-12b, qwen3vl-4b, qwen3vl-8b, flash-2.5, flash-2.5lite
# Example client names : "ollama/gemma3-4b", "transformers/gemma3-4b", "transformers/qwen3vl-4b", "vllm/gemma3-12b", "gemini/flash-2.5", "gemini_vtx/flash-2.5"

CLIENT_NAME = "ollama/gemma3-4b"        # "hosting/model", or "" to use experiment.json's active client
CONFIG_PATH = "configs/experiment.json"
DEBUG       = True                      # True = print resolved client params and request before running

IMAGE_PATHS = [
    "input/images/slide_020.png",
    "input/images/slide_021.png",
]
SYSTEM_PROMPT = ""
USER_PROMPT   = "Describe the two images and then summarize the main information shown."

SEPARATOR = "=" * 72


# ── Client resolution ────────────────────────────────────────────────────────
def get_client(cfg: Config, debug: bool = False) -> dict:
    """Resolve the target client config and optionally print its parameters."""
    client = cfg.get_client_by_name(CLIENT_NAME) if CLIENT_NAME else cfg.get_current_client()
    if debug:
        print(f"\n{SEPARATOR}")
        print(f"Hosting : {client['hosting']}")
        print(f"Client  : {client['name']}")
        print(f"Backend : {client['backend']}")
        print(f"Model   : {client['model_id']}")
        common_params = [
            f"max_tokens={client['max_tokens']}",
            f"temp={client['temperature']}",
            f"top_p={client['top_p']}",
        ]
        backend_params: list[str] = []

        if client["backend"] == "openai":
            print(f"Base URL: {client.get('base_url', 'https://api.openai.com/v1')}")
        elif client["backend"] == "gemini":
            backend_params.append(f"thinking_budget={client.get('thinking_budget')}")
        elif client["backend"] == "transformers":
            backend_params.append(f"quantization_level={client.get('quantization_level')}")

        all_params = common_params + backend_params
        print(f"Params  : {'  '.join(all_params)}")
        print(SEPARATOR)
    return client


# ── Payload construction ──────────────────────────────────────────────────────
def build_request_payload(
    system_prompt: str,
    prompt: str,
    image_paths: list[str],
    debug: bool = False,
) -> dict:
    """Build the content list (images + text) that will be sent to the backend."""
    if debug:
        print(f"Prompt : {prompt}")
        print(f"Images : {', '.join(image_paths)}")

    content = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        content.append(ImageBlock(str(path)))

    content.append(TextBlock(prompt))
    return {"content": content, "system_prompt": system_prompt}


# ── Inference ──────────────────────────────────────────────────────────────────
def run_client(client: dict, payload: dict) -> None:
    """Send the request to the backend, print the response and elapsed time."""
    request = InferenceRequest(
        content=payload["content"],
        system_prompt=payload["system_prompt"],
        max_new_tokens=client["max_tokens"],
        temperature=client["temperature"],
        top_p=client["top_p"],
    )

    backend = get_backend_from_config(client)

    start_time = time()
    resp = backend.run(request)
    elapsed = time() - start_time

    print(f"Status : OK ({elapsed:.2f}s)")
    print("Model output shown below")
    print(f"{SEPARATOR}")
    print(resp["text"].strip())
    if resp.get("logs"):
        print(f"\n--- Usage ---")
        for line in resp["logs"]:
            print(line)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(debug: bool = DEBUG) -> None:
    """Load config, build request payload, and run inference."""
    cfg = Config(CONFIG_PATH)
    client = get_client(cfg, debug=debug)
    payload = build_request_payload(SYSTEM_PROMPT, USER_PROMPT, IMAGE_PATHS, debug=debug)

    try:
        run_client(client, payload)
    except Exception as exc:
        print(f"\n{SEPARATOR}")
        print(f"Hosting : {client['hosting']}")
        print(f"Client : {client['name']}")
        print("Status : FAILED")
        print(f"Error  : {exc}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
