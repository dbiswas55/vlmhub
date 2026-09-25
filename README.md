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

An `ImageBlock` takes an image file (`image_path=`) or a PIL image in memory (`image=`), such as a video frame.

## Key Features

- **One interface, every backend** — a single `Model.generate()` call over `TextBlock` and `ImageBlock`, independent of model or backend.
- **Interleaved multimodal prompts** — mix `TextBlock` and `ImageBlock` in any order; each backend encodes as it needs (base64 data URI over the wire, PIL in-process).
- **A model registry** — models, settings and generation defaults live in one editable JSON file; give a model any short name you like alongside its real model id.
- **A hardware-aware Transformers backend** — `transformers` runs a model in-process with an optional 4-bit quantized load and a `fallback_dtype` for pre-Ampere GPUs (e.g. V100) that lack native bfloat16; [LiteLLM](https://docs.litellm.ai) covers every other hosting.

## Supported Backends

| Category | Provider | Hosting Key | Backend | How it runs |
|---|---|---|---|---|
| **Local Server** | Ollama | `ollama` | `litellm` | Local server on port 11434 |
| **Local Server** | MLX-VLM | `mlx_vlm` | `litellm` | Apple Silicon, port 8080 |
| **Local Server** | vLLM | `vllm` | `litellm` | CUDA GPU, port 8000 |
| **In-Process** | HuggingFace Transformers | `transformers` | `transformers` | Direct model loading (CUDA / MPS / CPU) |
| **Cloud API** | Google Gemini (AI Studio) | `gemini` | `litellm` | LiteLLM `gemini/` provider; `thinking_budget` (2.5) or `reasoning_effort` (3.x) |
| **Cloud API** | Google Gemini via Vertex AI | `vertex_ai` | `litellm` | LiteLLM `vertex_ai/`, Application Default Credentials |
| **Cloud API** | OpenAI | `openai` | `litellm` | GPT-4o, GPT-4o-mini, GPT-4.1 |
| **Cloud API** | Anthropic | `anthropic` | `litellm` | LiteLLM `anthropic/` provider |

Pre-configured models include **Gemma 3** (4B, 12B) and **Qwen3-VL** (4B, 8B) across all local hostings — plus Qwen2.5-VL, InternVL3.5, mPLUG-Owl3, Molmo2, Idefics3 and LLaVA-OneVision under `transformers` — and Gemini (2.5 / 3.x), GPT-4o/4.1, and Claude 3.x for cloud.

## Quick Start

### 1. Install

Pick whichever fits: **A** to try vlmhub out or work on it, **B** to install it as a library into another project.

#### A. Clone the repository

The repo carries the test scripts in [`tests/`](tests/), so this is the way to see it running end to end.

```bash
git clone https://github.com/dbiswas55/vlmhub.git
cd vlmhub

python3 -m venv venv312
source venv312/bin/activate      # macOS / Linux
# venv312\Scripts\activate       # Windows

pip install -e .
```

That covers every **API-hosted** backend — LiteLLM handles Gemini, Vertex AI, OpenAI, Anthropic
and all OpenAI-compatible local servers (Ollama, MLX-VLM, vLLM), so there are no per-hosting SDKs
to add. PyTorch is not installed: only the in-process `transformers` hosting needs it.

```bash
pip install -e ".[transformers]"   # torch + transformers, for in-process model loading
pip install -e ".[quantization]"   # adds bitsandbytes, for 4-bit loads under `transformers`
pip install -e ".[ollama]"         # only to manage Ollama's store via local_models.py
pip install -e ".[all]"            # everything above
```

#### B. Install into your own project

Add vlmhub as a submodule, then install it from there. The source sits in your project, readable and
upgradeable, and `pip` handles the dependencies and the import path.

```bash
cd ~/my-project
git init                          # skip if your project is already a git repository
git submodule add https://github.com/dbiswas55/vlmhub.git src/_libs/vlmhub

python3 -m venv venv312 && source venv312/bin/activate
pip install -e src/_libs/vlmhub                    # add "[transformers]" for the in-process hosting
```

Everything vlmhub brings stays inside that one folder:

```
my-project/
├── .env                    # your keys
└── src/_libs/vlmhub/       # the repository, pinned to one commit
```

Import it anywhere in your project — the editable install handles the path:

```python
from vlmhub import Model, TextBlock, ImageBlock

model = Model("gemini/flash-3.5")
```

**Upgrading** — move the submodule to the latest upstream commit:

```bash
git submodule update --remote --merge src/_libs/vlmhub
```

Then add and commit it as you would any other change — your repo records the new commit pointer, not
vlmhub's files. The next run picks up the new code; re-run `pip install -e src/_libs/vlmhub` only when
an upgrade adds a dependency.

### 2. Configure API Keys

```bash
cp .env.example .env                    # under A
cp src/_libs/vlmhub/.env.example .env   # under B
```

Then fill in only what you use — `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GCP_PROJECT`/`GCP_LOCATION`, `HF_TOKEN`/`HF_HOME`. A purely local Ollama or vLLM setup needs none of them. **Different variable names** — edit `api_key_env` in the registry instead.

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

### 4. Try the Example Scripts

The scripts in [`tests/`](tests/) are both smoke tests and usage examples — each downloads a few real
dataset samples, runs them through a client, and prints the output next to the ground truth. They
ship with the repository, so run them from a clone (**A**):

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
print(response["logs"])              # e.g. ["MODEL gemma3:4b", "USAGE prompt=288 output=304 total=592"]
print(response["model"])             # the model that actually served the request
print(response["params"])            # the generation settings actually sent, e.g. {"temperature": 0.3}
```

Generation parameters come from the registry (`defaults`, overridden per hosting, then per model) and can be overridden for one call:

```python
response = model.generate(content, max_new_tokens=1024, temperature=0.0, top_p=1.0,
                          top_k=40, reasoning_effort="medium")
```

A `None` value is not sent, so the provider's default applies (for Transformers, the model's own `generation_config`); `top_p`, `top_k` and `reasoning_effort` are `None` by default. `reasoning_effort` (`"low"`, `"medium"`, `"high"`, …) is mapped by LiteLLM to each provider's own control: effort on Claude, `thinking_level` on Gemini 3, `reasoning_effort` on OpenAI. Models marked `"sampling": false` in the registry reject `temperature`/`top_p`/`top_k` (e.g. Claude Sonnet 5, Opus 4.8+, GPT-5.6); those values are dropped for them, with a `DROPPED ...` line in `logs`, and `params` shows what was really sent. For reproducible runs, prefer dated model IDs where the provider offers them and record `model` and `params` with each result.

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
  "defaults": { "max_new_tokens": 4096, "temperature": 0.3, "top_p": null, "top_k": null, "reasoning_effort": null },
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

**Editing it in place** works for a quick change. Git keeps your edits, but they land in vlmhub's
history rather than your project's, and a release that also touches `models.json` blocks the next
`git submodule update` until you stash or commit them.

**Extending the registry** — for anything you mean to keep, pass your own file instead. It holds only
what you add or change:

```jsonc
// my_models.json
{
  "active": { "hosting": "my_server", "model": "my-finetune" },
  "defaults": { "temperature": 0.0 },
  "hostings": {
    "my_server": {
      "backend": "litellm",
      "litellm_prefix": "openai",
      "api_base": "http://localhost:8000/v1",
      "models": [{ "name": "my-finetune", "model_id": "org/my-finetuned-vlm" }]
    }
  }
}
```

```python
model = Model("my_server/my-finetune", models_path="my_models.json")
```

It has the same shape as the bundled file and is deep-merged over it, so you can add hostings and models or override single fields. One catch: lists are replaced, not appended — naming an existing hosting's `models` drops the rest, so add a new hosting key instead. See [Adding a New Model](src/vlmhub/backends/README.md#adding-a-new-model) and [Adding a New Hosting](src/vlmhub/backends/README.md#adding-a-new-hosting).

**Transformers models** take extra per-model fields — `model_class` and `fallback_dtype`, plus an optional `processor_kwargs` — described in [the backend guide](src/vlmhub/backends/README.md#transformers).

## Local Model Management

[`src/vlmhub/utils/local_models.py`](src/vlmhub/utils/local_models.py) manages both the HuggingFace cache and Ollama's store:

```bash
python -m vlmhub.utils.local_models
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

## License

Released under the MIT License — see [LICENSE](LICENSE).
