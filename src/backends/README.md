# Backend Setup

This project can talk to a model through a cloud API or something running on your own machine, all through one interface — pick what fits your setup below and skip the rest.

## Which one do you need?

| Your situation | Use | See |
|---|---|---|
| No local install, just an API key | `gemini` or `openai` | Cloud APIs |
| Local models on macOS | `ollama` (easiest) or `mlx_vlm` (faster on Apple Silicon) | Local Models |
| Local models on CUDA / Linux | `vllm` (server, optimized for throughput) or `transformers` (in-process, no server) | Local Models |
| Want to list, download/pull, or delete local models | — | Local Model Management |

## Python Environment

Needed regardless of which option above you pick. Create and activate a virtual environment:

```bash
python3 -m venv venv312
source venv312/bin/activate                  # macOS / Linux
venv312\Scripts\activate                     # Windows
```

Install all dependencies:

```bash
pip install mlx mlx-vlm torch torchvision Pillow transformers accelerate \
            huggingface_hub python-dotenv openai google-genai ollama
```

Configure environment — create a `.env` file at the project root:

```env
HF_TOKEN=hf_your_token_here    # huggingface.co/settings/tokens
HF_HOME=.cache/huggingface     # optional; use absolute path on a cluster
GEMINI_API_KEY=...             # for the gemini / gemini_oai hostings
OPENAI_API_KEY=...             # for the openai hosting
```

## Local Model Management

Skip this if you're only using Cloud APIs. [`model_cache.py`](model_cache.py) manages both local model stores — the HuggingFace cache (used by MLX-VLM, vLLM, and Transformers, which all load models by HF repo id) and Ollama's own store. The two are independent directories with independent formats; the script gives them a matching set of `list_*` / `download_*` / `delete_*` functions.

Edit the `hf_models` / `ollama_models` lists at the bottom of the file, then run it directly to inspect, download into, and prune both stores:

```bash
python src/backends/model_cache.py
```

## Cloud APIs

No server, no local model files — just an API key (above) and a hosting choice in `configs/experiment.json`:

- **`gemini`** — native SDK via Google AI Studio. Simplest option, supports `thinking_budget`.
- **`gemini_vtx`** — same native SDK, routed through Vertex AI (needs `GCP_PROJECT`/`GCP_LOCATION` and Application Default Credentials).
- **`gemini_oai`** — Gemini via the OpenAI-compatible endpoint, for one client style across providers. No `thinking_budget` control here.
- **`openai`** — OpenAI's own API.

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
# API base: http://localhost:8000/v1  (OpenAI-compatible)
```

Browse models: https://docs.vllm.ai/en/latest/models/supported_models.html

### Transformers

Loads a model directly into the Python process — no server to install or run, just a model in the HF cache (see Local Model Management above). Device (CUDA / MPS / CPU) is auto-detected; these fields in `configs/experiment.json` control how it loads:

- **`fallback_dtype`** (per model) — every model here is native bfloat16, so that is the load dtype wherever the hardware supports it. Pre-Ampere GPUs (e.g. V100) don't, and this field decides what they use instead: `"float16"` where it has been verified safe, or `null` to keep bfloat16 and let the GPU emulate it — correct but slow. fp16 is not a universal substitute: bf16-trained models like Gemma-3 overflow it into NaN logits.
- **`quantization_level`** (per hosting) — never automatic. `"4bit"` loads 4-bit NF4 on CUDA, ignored with a warning on MPS/CPU; `null` loads full precision. Being hosting-wide, it applies to every model in the run — reach for it when a model's weights won't fit the GPU.
- **`model_class`** (per model, optional) — the transformers auto-class to load with; omitted means `AutoModelForImageTextToText`.
- **`processor_kwargs`** (per model, optional) — extra arguments passed as-is to `AutoProcessor.from_pretrained`.

On CUDA, weights are placed with `device_map="auto"`, so a model too large for one card spreads across every visible GPU; set `CUDA_VISIBLE_DEVICES=0` to pin it to one.

## Adding a New Model

A new model needs no code changes, just a config entry — code changes are only for a new *backend* (new protocol), see the root [README](../../README.md#adding-a-new-backend).

1. Pick the hosting it belongs to under `models.hostings` in `configs/experiment.json` (e.g. `ollama`, `gemini`, `transformers`).
2. Append it to that hosting's `models` list: `{ "name": "<short-alias>", "model_id": "<the model's real id>" }`. `model_id` must be exact — the provider's API name (`gemini`/`openai`), Ollama's tag, or the HF repo id (`mlx_vlm`/`vllm`/`transformers`).
3. For a local hosting, make sure the model is actually downloaded first: `ollama pull <tag>` for Ollama, or `download_hf_model(...)` / `download_ollama_model(...)` in [`model_cache.py`](model_cache.py) (Local Model Management above) for everything else.
4. Reference it in code as `"hosting/name"`, e.g. `Config().get_client_by_name("ollama/my-new-model")`.

## Under the Hood

For anyone extending this rather than just using it: a **backend** is the protocol (`gemini` native SDK, `openai`-compatible chat completions, or in-process `transformers`), implemented in [`backends.py`](backends.py). A **hosting** is a named entry in `configs/experiment.json` pairing a backend with connection details and a model list — several hostings can share one backend (`openai`, `ollama`, `mlx_vlm`, `vllm` are all the OpenAI-compatible backend, just pointed at different servers).

- [`src/utils/config.py`](../utils/config.py)'s `Config` class loads `configs/experiment.json` and resolves a `"hosting/model"` name into a merged client dict.
- [`__init__.py`](__init__.py) turns that dict into a running backend instance.
- [`request.py`](request.py) defines the request/content types (`TextBlock`, `ImageBlock`, etc.) a backend accepts.
