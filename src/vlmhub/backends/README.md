# Backend Setup

vlmhub can talk to a model through a cloud API or something running on your own machine, all through one interface — pick what fits your setup below and skip the rest.

## Which one do you need?

| Your situation | Use | See |
|---|---|---|
| No local install, just an API key | `gemini`, `openai`, or `anthropic` | Cloud APIs |
| Local models on macOS | `ollama` (easiest) or `mlx_vlm` (faster on Apple Silicon) | Local Models |
| Local models on CUDA / Linux | `vllm` (server, optimized for throughput) or `transformers` (in-process, no server) | Local Models |
| Want to list, download/pull, or delete local models | — | Local Model Management |

## Python Environment

Create and activate a virtual environment, then install the package:

```bash
python3 -m venv venv312
source venv312/bin/activate                  # macOS / Linux
venv312\Scripts\activate                     # Windows

pip install -e .
```

That covers every hosting: there are no per-hosting client SDKs to add, because
[LiteLLM](https://docs.litellm.ai) speaks to Gemini, Vertex AI, OpenAI, Anthropic
and every OpenAI-compatible local server through one package, and it is a core
dependency.

Two extras exist for things not everyone needs, plus `all` to get both:

| Extra | Install | What it is for |
|---|---|---|
| `quantization` | `pip install -e ".[quantization]"` | `bitsandbytes`, for `quantization_level: "4bit"` on CUDA under the `transformers` hosting |
| `ollama` | `pip install -e ".[ollama]"` | the `ollama` Python package, needed only to list/download/delete models in Ollama's own store via `local_models.py` — **not** to call a running Ollama server |
| `all` | `pip install -e ".[all]"` | both of the above |

`mlx`/`mlx-vlm` and `vllm` are **not** client-side packages — they're only for the machine that *serves* a model that way (`python -m mlx_vlm.server` / `vllm serve`, see below). A client pointed at either server needs nothing extra.

Configure environment — copy [`.env.example`](../../../.env.example) to `.env` at the project root and fill in what you use:

```env
HF_TOKEN=hf_your_token_here    # huggingface.co/settings/tokens
HF_HOME=.cache/huggingface     # optional; use absolute path on a cluster
GEMINI_API_KEY=...             # for the gemini hosting
OPENAI_API_KEY=...             # for the openai hosting
ANTHROPIC_API_KEY=...          # for the anthropic hosting
GCP_PROJECT=...                # for the vertex_ai hosting
GCP_LOCATION=...               # for the vertex_ai hosting
```

## Local Model Management

Skip this if you're only using Cloud APIs. [`local_models.py`](../utils/local_models.py) manages both local model stores — the HuggingFace cache (used by MLX-VLM, vLLM, and Transformers, which all load models by HF repo id) and Ollama's own store. The two are independent directories with independent formats; the script gives them a matching set of `list_*` / `download_*` / `delete_*` functions.

Edit the `hf_models` / `ollama_models` lists at the bottom of the file, then run it directly to inspect, download into, and prune both stores:

```bash
python src/vlmhub/utils/local_models.py
```

## Cloud APIs

No server, no local model files — just an API key (above) and a hosting choice in [`models.json`](../models.json):

- **`gemini`** — Gemini via Google AI Studio, on LiteLLM's native `gemini/` provider. Simplest option, and supports `thinking_budget`.
- **`vertex_ai`** — the same Gemini models routed through Vertex AI (needs `GCP_PROJECT`/`GCP_LOCATION` and Application Default Credentials).
- **`openai`** — OpenAI's own API.
- **`anthropic`** — Anthropic's own API, on LiteLLM's native `anthropic/` provider.

## Local Models

Everything below runs on your own machine:

- **`ollama`** — easiest to set up; good default for local use.
- **`mlx_vlm`** — macOS / Apple Silicon only; faster than Ollama on Apple hardware.
- **`vllm`** — CUDA / Linux; a server optimized for high-throughput / concurrent requests.
- **`transformers`** — CUDA / Linux; in-process, no server to run at all.

All four need a model downloaded first — see Local Model Management above.

### Ollama

Install:

```bash
brew install ollama                              # macOS
curl -fsSL https://ollama.com/install.sh | sh    # Linux
```

Pull a model and start the server:

```bash
ollama pull gemma3:4b
ollama serve                    # API base: http://localhost:11434/v1
```

Useful commands:

```bash
ollama list                     # list downloaded models
ollama rm gemma3:4b             # remove a model
```

Browse models: https://ollama.com/library

### MLX-VLM

Download a model into the HF cache (see Local Model Management above), then serve:

```bash
source venv312/bin/activate     # the environment where mlx-vlm is installed
python -m mlx_vlm.server --model mlx-community/gemma-3-4b-it-qat-4bit --port 8080
# API base: http://localhost:8080/v1  (OpenAI-compatible)
```

Browse available MLX models:
- https://huggingface.co/mlx-community
- Gemma3: https://huggingface.co/collections/mlx-community/gemma-3-qat
- Qwen3-VL: https://huggingface.co/collections/mlx-community/qwen3-vl

### vLLM

Install:

```bash
pip install vllm
```

Serve a model:

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
# API base: http://127.0.0.1:8000/v1  (OpenAI-compatible)
```

Browse models: https://docs.vllm.ai/en/latest/models/supported_models.html

### Transformers

Loads a model directly into the Python process — no server to install or run, just a model in the HF cache (see Local Model Management above). Device (CUDA / MPS / CPU) is auto-detected; these fields in [`models.json`](../models.json) control how it loads:

- **`fallback_dtype`** (per model) — every model here is native bfloat16, so that is the load dtype wherever the hardware supports it. Pre-Ampere GPUs (e.g. V100) don't, and this field decides what they use instead: `"float16"` where it has been verified safe, or `null` to keep bfloat16 and let the GPU emulate it — correct but slow. fp16 is not a universal substitute: bf16-trained models like Gemma-3 overflow it into NaN logits.
- **`quantization_level`** (per hosting) — never automatic. `"4bit"` loads 4-bit NF4 on CUDA, ignored with a warning on MPS/CPU; `null` loads full precision. Being hosting-wide, it applies to every model in the run — reach for it when a model's weights won't fit the GPU. Needs the `quantization` extra.
- **`model_class`** (per model, required) — the transformers auto-class to load with: `AutoModelForImageTextToText` for most VLMs, or whatever class the model is registered under (mPLUG-Owl3 is an `AutoModelForCausalLM`). Loading fails fast if it is missing or misspelled.
- **`processor_kwargs`** (per model, optional) — extra arguments passed as-is to `AutoProcessor.from_pretrained`.

On CUDA, weights are placed with `device_map="auto"`, so a model too large for one card spreads across every visible GPU; set `CUDA_VISIBLE_DEVICES=0` to pin it to one.

## Adding a New Model

A new model needs no code changes, just a registry entry.

1. Pick the hosting it belongs to under `hostings` in [`models.json`](../models.json) (e.g. `ollama`, `gemini`, `transformers`).
2. Append it to that hosting's `models` list: `{ "name": "<short-alias>", "model_id": "<the model's real id>" }`. `model_id` must be exact — the provider's API name (`gemini`/`openai`), Ollama's tag, or the HF repo id (`mlx_vlm`/`vllm`/`transformers`).
3. For a local hosting, make sure the model is actually downloaded first: `ollama pull <tag>` for Ollama, or `download_hf_model(...)` / `download_ollama_model(...)` in [`local_models.py`](../utils/local_models.py) (Local Model Management above) for everything else.
4. Reference it in code as `"hosting/name"`, e.g. `Model("ollama/my-new-model")`.

To add models without editing the packaged registry, put them in your own JSON file with the same shape and pass it through: `Model("ollama/my-new-model", models_path="my_models.json")`. It is deep-merged over the bundled registry.

## Adding a New Hosting

A **hosting** is a named entry in `models.json` pairing a backend with connection details and a model list. Most new ones are **zero code** — LiteLLM already speaks the protocol, so the hosting just names a `litellm_prefix` and, where relevant, an `api_base` and credentials:

```jsonc
"my_server": {
  "backend": "litellm",
  "litellm_prefix": "openai",              // LiteLLM provider prefix
  "api_base": "http://localhost:9000/v1",  // omit for a hosted provider
  "models": [ { "name": "my-model", "model_id": "org/my-model" } ]
}
```

The factory joins `litellm_prefix` and `model_id` into LiteLLM's `"<provider>/<model>"` routing string, so any of [LiteLLM's providers](https://docs.litellm.ai/docs/providers) works here. Several hostings can share one backend — `openai`, `ollama`, `mlx_vlm` and `vllm` are all `litellm`, just pointed at different servers.

For credentials: a real secret goes in `api_key_env` (naming an env var, e.g. `"MY_API_KEY"`, filled in your `.env`). An `litellm_prefix: "openai"` hosting with a custom `api_base` and no `api_key_env` — a local OpenAI-compatible server, like `ollama`/`mlx_vlm` above — gets a harmless placeholder key automatically, since those servers want *a* token but don't check it.

Recognized hosting keys: `litellm_prefix`, `api_base`, `api_key_env`, `thinking_budget`, `vertex_project_env`, `vertex_location_env`.

## Adding a New Backend

Only needed for something LiteLLM cannot reach at all — a genuinely different execution model, the way `transformers` runs in-process rather than over a wire.

1. Subclass `BaseBackend` in [`backends.py`](backends.py) and implement `run(request) -> dict`, returning `{"text": str, "logs": list[str]}`.
2. Register it in [`__init__.py`](__init__.py): add a branch to `get_backend_from_config` matching on a new `"backend"` value, mapping config keys to constructor arguments.
3. Add a hosting in [`models.json`](../models.json) with `"backend": "<your new value>"`.

## Under the Hood

A **backend** is *how* a request is executed — `litellm` (any API-hosted model) or `transformers` (in-process) — implemented in [`backends.py`](backends.py). A **hosting** is *where* it goes, defined in config.

- [`config.py`](../utils/config.py)'s `Config` class loads `models.json` and resolves a `"hosting/model"` name into a merged client dict.
- [`__init__.py`](__init__.py) turns that dict into a running backend instance.
- [`request.py`](request.py) defines the request/content types (`TextBlock`, `ImageBlock`, `InferenceRequest`) a backend accepts, and `InferenceRequest.to_openai_messages()`, which every API-hosted backend builds its payload with.
- [`model.py`](../model.py)'s `Model` ties the two together — the class you actually use.
