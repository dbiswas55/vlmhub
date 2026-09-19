# vlmhub

A lightweight, config-driven framework for **unified vision-language model inference** across local and cloud backends. Run multimodal prompts — interleaved text and images — against Ollama, MLX-VLM, vLLM, HuggingFace Transformers, Gemini, Vertex AI, OpenAI, or Anthropic without rewriting any inference code.

```python
from vlmhub import Model, TextBlock, ImageBlock

model = Model("ollama/gemma3-4b")            # local server
# model = Model("gemini/flash-3.5lite")      # cloud API
# model = Model("transformers/qwen3vl-8b")   # in-process, no server

reply = model.generate([
    TextBlock("Here are two images:"),
    ImageBlock(image_path="image1.png"),     # an image file on disk
    ImageBlock(image=pil_image),             # or a PIL image in memory
    TextBlock("What are the main differences between them?"),
])
print(reply["text"])
```

Switching between them is the same two lines with a different client name.

An `ImageBlock` takes an image file (`image_path=`) or a PIL image in memory (`image=`), such as a video frame.

## Key Features

- **One interface, every backend** — a single `Model.generate()` call over `TextBlock` and `ImageBlock`, independent of which model or backend is behind it.
- **Interleaved multimodal prompts** — build prompts by mixing `TextBlock` and `ImageBlock` in any order; each backend handles encoding as it needs (base64 data URI over the wire, PIL in-process).
- **A model registry** — models, settings and generation defaults live in one JSON file, editable in place; give a model any short name you like alongside its real model id.
- **Transformers backend, hardware-aware** — [LiteLLM](https://docs.litellm.ai) covers 7 of the 8 hostings, while the `transformers` backend runs a model in-process with an optional 4-bit quantized load, and a `fallback_dtype` for pre-Ampere GPUs (e.g. V100) that lack native bfloat16.

## Supported Backends

| Category | Provider | Hosting Key | Backend | How it runs |
|---|---|---|---|---|
| **Local Server** | Ollama | `ollama` | `litellm` | Local server on port 11434 |
| **Local Server** | MLX-VLM | `mlx_vlm` | `litellm` | Apple Silicon, port 8080 |
| **Local Server** | vLLM | `vllm` | `litellm` | CUDA GPU, port 8000 |
| **In-Process** | HuggingFace Transformers | `transformers` | `transformers` | Direct model loading (CUDA / MPS / CPU) |
| **Cloud API** | Google Gemini (AI Studio) | `gemini` | `litellm` | LiteLLM `gemini/` provider; supports `thinking_budget` |
| **Cloud API** | Google Gemini via Vertex AI | `vertex_ai` | `litellm` | LiteLLM `vertex_ai/`, Application Default Credentials |
| **Cloud API** | OpenAI | `openai` | `litellm` | GPT-4o, GPT-4o-mini, GPT-4.1 |
| **Cloud API** | Anthropic | `anthropic` | `litellm` | LiteLLM `anthropic/` provider |

Pre-configured models include **Gemma 3** (4B, 12B) and **Qwen3-VL** (4B, 8B) across all local hostings — plus Qwen2.5-VL, InternVL3.5, mPLUG-Owl3, Molmo2, Idefics3 and LLaVA-OneVision under `transformers` — and Gemini (2.5 / 3.x), GPT-4o/4.1, and Claude 3.x for cloud.

## Quick Start

### 1. Create Environment and Install

```bash
python3 -m venv venv312
source venv312/bin/activate      # macOS / Linux
# venv312\Scripts\activate       # Windows

pip install -e .
```

That covers every hosting — LiteLLM handles Gemini, Vertex AI, OpenAI, Anthropic and all
OpenAI-compatible local servers, so there are no per-hosting SDKs to add. Three
optional extras:

```bash
pip install -e ".[quantization]"   # bitsandbytes, for 4-bit loads under `transformers`
pip install -e ".[ollama]"         # only to manage Ollama's store via local_models.py
pip install -e ".[samples]"        # pyarrow, only for the example scripts in tests/
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Then fill in only what you use — `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GCP_PROJECT`/`GCP_LOCATION`, `HF_TOKEN`/`HF_HOME`. A purely local Ollama or vLLM setup needs none of them.

### 3. Set Up a Local Backend (Optional)

**Ollama** (easiest to start with):

```bash
brew install ollama                # macOS
ollama pull gemma3:4b
ollama serve                       # http://localhost:11434/v1
```

**MLX-VLM** (Apple Silicon):

```bash
python -m mlx_vlm.server --model mlx-community/gemma-3-4b-it-qat-4bit --port 8080
```

**vLLM** (CUDA):

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

See [`src/vlmhub/backends/README.md`](src/vlmhub/backends/README.md) for the full setup guide, and [`src/vlmhub/utils/local_models.py`](src/vlmhub/utils/local_models.py) for model download utilities.

### 4. Run Inference

The scripts in [`tests/`](tests/) are both smoke tests and usage examples — each downloads a few real
dataset samples, runs them through a client, and prints the output next to the ground truth.
They need the `samples` extra (`pip install -e ".[samples]"`):

```bash
python tests/test_captioning.py                                 # models.json's active client
python tests/test_vqa.py --client ollama/gemma3-4b
python tests/test_vqa.py --client gemini/flash-2.5 --n 5
python tests/test_captioning.py --client transformers/gemma3-4b
```

## Usage

`Model` is the whole surface: it resolves a client from the registry and holds the backend that serves it.

```python
from vlmhub import Model, TextBlock, ImageBlock

model = Model()                      # models.json's "active" client
model = Model("ollama/gemma3-4b")    # a specific one, as "hosting/model"

response = model.generate(
    [
        ImageBlock(image_path="input/images/slide_1.png"),
        TextBlock("What does this diagram show?"),
        ImageBlock(image_path="input/images/slide_2.png"),
        TextBlock("How does this compare to the previous slide?"),
    ],
    system_prompt="You are a helpful assistant.",
)

print(response["text"])              # the generation
print(response["logs"])              # e.g. ["USAGE prompt=288 output=304 total=592"]
```

Generation parameters come from the registry (`defaults`, overridden per hosting, then per model) and can be overridden for one call:

```python
response = model.generate(content, max_new_tokens=1024, temperature=0.0, top_p=1.0)
```

Useful properties: `model.name` (`"ollama/gemma3-4b"`), `model.model_id` (`"gemma3:4b"`), `model.short_name` (`"gemma3-4b"`), and `model.report()` to print both.

Images are encoded automatically — base64 data URI for every API-hosted backend, PIL for Transformers. An `ImageBlock(image=...)` is encoded as PNG, so nothing is re-compressed on the way out.

## Project Structure

```
vlmhub/
├── pyproject.toml
├── .env.example                 # API keys and HF token (copy to .env)
├── src/vlmhub/
│   ├── __init__.py              # Public API: Model, Config, TextBlock, ImageBlock, …
│   ├── models.json              # The model/hosting registry
│   ├── model.py                 # Model — a client's config plus its backend
│   ├── utils/
│   │   ├── config.py            # Config — resolves "hosting/model" into a client dict
│   │   └── local_models.py      # List / download / delete HF-cache and Ollama models
│   └── backends/
│       ├── __init__.py          # Backend factory (get_backend_from_config)
│       ├── backends.py          # BaseBackend, LiteLLMBackend, TransformersBackend
│       ├── request.py           # TextBlock, ImageBlock, InferenceRequest
│       └── README.md            # Backend setup guide
└── tests/
    ├── test_vqa.py              # VQAv2 samples — image + questions vs. ground-truth answers
    └── test_captioning.py       # COCO samples — image + generated vs. reference captions
```

## Configuration

The model registry lives in [`src/vlmhub/models.json`](src/vlmhub/models.json), which ships with the package:

```jsonc
{
  "active": { "hosting": "ollama", "model": "gemma3-4b" },      // default client
  "defaults": { "max_new_tokens": 4096, "temperature": 0.3, "top_p": 1.0 },
  "hostings": {
    "ollama": {
      "backend": "litellm",
      "litellm_prefix": "openai",
      "api_base": "http://localhost:11434/v1",
      "models": [
        { "name": "gemma3-4b", "model_id": "gemma3:4b" }
        // ...
      ]
    }
    // gemini, vertex_ai, openai, anthropic, mlx_vlm, vllm, transformers ...
  }
}
```

**Selecting a client** — pass `"hosting/model"` to `Model(...)`, or set `active` in the registry and call `Model()`. Model-level fields override hosting-level fields, which override `defaults`.

**Extending the registry** — rather than editing the packaged file, pass your own:

```python
model = Model("my_server/my-model", models_path="my_models.json")
```

Your file has the same flat shape and is deep-merged over the bundled one, so you can add hostings and models or override individual fields. Lists are replaced wholesale: naming an existing hosting's `models` replaces that hosting's models rather than appending, so add a new hosting key to extend. See [Adding a New Model](src/vlmhub/backends/README.md#adding-a-new-model) and [Adding a New Hosting](src/vlmhub/backends/README.md#adding-a-new-hosting).

**Transformers models** take extra per-model fields — `model_class` and `fallback_dtype`, plus an optional `processor_kwargs` — described in [the backend guide](src/vlmhub/backends/README.md#transformers).

## Local Model Management

[`src/vlmhub/utils/local_models.py`](src/vlmhub/utils/local_models.py) manages both the HuggingFace cache and Ollama's store:

```bash
python src/vlmhub/utils/local_models.py
```

| Function | Description |
|---|---|
| `list_hf_cache_models()` / `list_ollama_cache_models()` | List models with sizes |
| `download_hf_model(ids)` / `download_ollama_model(ids)` | Download one model or a list |
| `delete_hf_cache_model(id)` / `delete_ollama_cache_model(id)` | Delete a specific model |
| `delete_hf_cache_model_interactive()` / `delete_ollama_cache_model_interactive()` | Interactive picker to delete models |

The Ollama functions need the `ollama` extra; the HuggingFace ones work out of the box.

## Adding a New Backend

Most new providers need **no code at all** — add a hosting to `models.json` naming a [LiteLLM provider prefix](https://docs.litellm.ai/docs/providers), and the existing `LiteLLMBackend` serves it. See [Adding a New Hosting](src/vlmhub/backends/README.md#adding-a-new-hosting).

A genuinely new *backend type* — something LiteLLM cannot reach, the way in-process Transformers isn't a wire protocol — means subclassing `BaseBackend` in [`backends.py`](src/vlmhub/backends/backends.py) and registering it in [`get_backend_from_config`](src/vlmhub/backends/__init__.py). See [Adding a New Backend](src/vlmhub/backends/README.md#adding-a-new-backend).

## Scope

vlmhub is about **generation and hosting only** — turning a request into a response, across whichever backend you point it at. Datasets, tasks, prompt templates and multi-step workflows deliberately live outside it, in whatever project consumes this one.

## License

This project is open source. See [LICENSE](LICENSE) for details.
