"""All inference backends — BaseBackend, LiteLLMBackend, TransformersBackend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image
from .request import ImageBlock, InferenceRequest, TextBlock

# Model ID substrings that use the inline-image content style
# (image dict embedded inside the content list rather than passed via images= kwarg).
# Add a substring here to support any new model family without touching run() logic.
_INLINE_IMAGE_MODEL_PATTERNS: frozenset[str] = frozenset([
    "qwen", "internvl",      # Qwen2-VL, Qwen3-VL, and future Qwen VL variants
])

# The transformers auto-class a model loads through. Image-text VLMs are served by
# AutoModelForImageTextToText; a model registered under a different class — mPLUG-Owl3
# is an AutoModelForCausalLM — names that class as "model_class" in its config entry.
_DEFAULT_MODEL_CLASS = "AutoModelForImageTextToText"

# The schemes a "quantization_level" may name; None is full precision. Listing one here
# only lets config carry it — _build_quantization_config() decides what it can build, and
# refuses at load time anything it has no branch for.
_QUANTIZATION_LEVELS: tuple[str | None, ...] = (None, "4bit")

# The dtypes a model's "fallback_dtype" may name: what to load with on a GPU that has no
# native bfloat16. None means no substitute — stay in bfloat16 and let it be emulated.
_FALLBACK_DTYPES: dict[str | None, "torch.dtype | None"] = {
    None: None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseBackend(ABC):
    """Common interface for all inference backends."""

    name: str

    @abstractmethod
    def run(self, request: InferenceRequest) -> dict:
        """Run inference and return ``{"text": str, "logs": list[str]}``."""
        ...


# ── LiteLLM (any API-hosted model) ────────────────────────────────────────────

class LiteLLMBackend(BaseBackend):
    """Any API-hosted model, routed through LiteLLM's unified completion().

    Covers what GeminiBackend and OpenAIBackend used to split between:
    native Gemini/Vertex AI, OpenAI, and any OpenAI-compatible server
    (Ollama, MLX-VLM, vLLM) — one class, config decides which.
    """

    def __init__(
        self,
        name: str,
        litellm_model: str,           # e.g. "vertex_ai/gemini-3.1-pro-preview",
                                      #      "hosted_vllm/google/gemma-3-12b-it",
                                      #      "openai/gemma3:4b"
        api_base: str | None = None,
        api_key: str | None = None,
        thinking_budget: int | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ):
        self.name = name
        self.litellm_model = litellm_model
        self._api_base = api_base
        self._api_key = api_key
        self._thinking_budget = thinking_budget
        self._vertex_project = vertex_project
        self._vertex_location = vertex_location

    def run(self, request: InferenceRequest) -> dict:
        """Generate text output via LiteLLM's unified completion API."""
        from litellm import completion

        kwargs: dict = {
            "model": self.litellm_model,
            "messages": request.to_openai_messages(),
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._vertex_project:
            kwargs["vertex_project"] = self._vertex_project
        if self._vertex_location:
            kwargs["vertex_location"] = self._vertex_location
        if request.max_new_tokens is not None:
            kwargs["max_tokens"] = request.max_new_tokens
        if self._thinking_budget is not None:
            # Native param, not reasoning_effort — sidesteps mapping bugs
            # seen on some vertex_ai/gemini model+version combinations.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}

        response = completion(**kwargs)
        logs: list[str] = []
        usage = getattr(response, "usage", None)
        if usage is not None:
            logs.append(f"USAGE prompt={usage.prompt_tokens} "
                        f"output={usage.completion_tokens} "
                        f"total={usage.total_tokens}")
        return {"text": response.choices[0].message.content, "logs": logs}


# ── Transformers (in-process) ─────────────────────────────────────────────────

class TransformersBackend(BaseBackend):
    """Local in-process inference via HuggingFace Transformers.

    A model is described entirely by its entry under the "transformers" hosting in
    models.json. Four of those fields shape how it loads:

      model_class         the transformers auto-class to load through, defaulting
                          to AutoModelForImageTextToText.
      fallback_dtype      the dtype to use on a GPU without native bfloat16.
      quantization_level  None for full precision, "4bit" for 4-bit NF4 on CUDA.
      processor_kwargs    extra arguments passed as-is to AutoProcessor.from_pretrained,
                          e.g. {"do_image_splitting": false} for Idefics3.

    On CUDA the weights are placed by accelerate across every visible GPU, so a model
    too large for one card still loads; quantized or not makes no difference to that.

    The device (CUDA > MPS > CPU) and the load dtype are settled in __init__; the
    quantization scheme and the auto-class are checked by the methods that own them,
    on the first run() call but still before anything is downloaded. The weights
    themselves load lazily, on that same first call.
    """

    def __init__(
        self,
        name: str,
        hf_model_id: str,
        hf_token: str | None = None,
        hf_cache: str | None = None,
        quantization_level: str | None = None,
        fallback_dtype: str | None = None,
        model_class: str | None = None,
        processor_kwargs: dict | None = None,
    ):
        if not hf_model_id:
            raise ValueError(f"[{name}] hf_model_id is required for Transformers backend.")
        self.name = name
        self.hf_model_id = hf_model_id
        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.quantization_level = self._checked("quantization_level", quantization_level, _QUANTIZATION_LEVELS)
        # Consulted only on a pre-Ampere GPU; see _pick_compute_dtype().
        self.fallback_dtype = self._checked("fallback_dtype", fallback_dtype, _FALLBACK_DTYPES)
        # Config names a class only where it differs from the default.
        self.model_class = model_class or _DEFAULT_MODEL_CLASS
        # Not interpreted here: each key is the processor's own setting.
        self.processor_kwargs = processor_kwargs or {}

        self.device = self._pick_device()
        self._model = None
        self._processor = None
        self.compute_dtype = self._pick_compute_dtype()

    # ── Config → load decisions (all resolved before any download) ───────────

    def _checked(self, field: str, value: str | None, allowed) -> str | None:
        """Lower-case a config value and check it names something this backend knows."""
        value = value.lower() if value else None
        if value not in allowed:
            raise ValueError(f"[{self.name}] {field} must be one of: "
                             f"{', '.join(repr(v) for v in allowed)}. Got: {value!r}")
        return value

    @staticmethod
    def _pick_device() -> str:
        """Detect best available device: CUDA > MPS > CPU."""

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_compute_dtype(self):
        """Resolve the dtype to load the weights in.

        Every model served here is natively bfloat16: that is the dtype wherever
        the hardware provides it, float32 on CPU. Pre-Ampere GPUs (V100 = sm_70)
        have no bfloat16 units, so there `fallback_dtype` picks the lesser evil —
        "float16", fast but narrow enough in exponent range that bf16-trained
        models like Gemma-3 overflow into NaN logits, or None to stay in bfloat16
        and let PyTorch emulate it: exact, same 2 bytes/param, but slow.
        """

        if self.device == "cpu":
            return torch.float32
        if self.device == "mps":
            # Safe on M1+ Macs; use float16 instead on older/unsupported chips.
            return torch.bfloat16

        major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major >= 8:                                  # Ampere / Ada / Hopper
            return torch.bfloat16

        print(
            f"[{self.name}] GPU {torch.cuda.get_device_name()} (sm_{major}x) has no native "
            f"bfloat16; using {self.fallback_dtype or 'emulated bfloat16'}."
        )
        return _FALLBACK_DTYPES[self.fallback_dtype] or torch.bfloat16

    def _build_quantization_config(self):
        """Build the bitsandbytes config for a quantized load, or None for a full one.

        Quantization is opt-in, and applies where `quantization_level` is "4bit"
        and the device is CUDA. bitsandbytes has no kernels for MPS or CPU, so a
        request there is reported and dropped rather than raised. Quantized
        weights compute in whatever dtype _pick_compute_dtype() settled on.
        Any other level raises: a scheme this method cannot build must not load
        silently unquantized, which would make a run mean something else.
        """
        if self.quantization_level is None:
            return None

        if self.device != "cuda":
            print(
                f"[{self.name}] quantization_level={self.quantization_level} requested, "
                f"but device '{self.device}' does not support bitsandbytes quantization. "
                "Falling back to non-quantized loading."
            )
            return None

        from transformers import BitsAndBytesConfig

        if self.quantization_level == "4bit":
            print(f"Configuration: 4-bit (NF4) | Compute dtype: {self.compute_dtype}")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        raise NotImplementedError(
            f"[{self.name}] quantization_level={self.quantization_level!r} is not "
            f"implemented; only '4bit' (NF4) is supported. Add a branch above with its "
            f"BitsAndBytesConfig to enable it."
        )

    def _resolve_model_class(self):
        """Look up this model's transformers auto-class by name."""
        import transformers

        cls = getattr(transformers, self.model_class, None)
        if cls is None:
            raise ValueError(
                f"[{self.name}] model_class '{self.model_class}' is not a class in the "
                f"installed transformers ({transformers.__version__}). Check the spelling "
                f"in models.json, or omit it to use {_DEFAULT_MODEL_CLASS}."
            )
        return cls

    # ── Load and run ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load processor and model on first use."""
        if self._model is not None:
            return

        from transformers import AutoProcessor

        model_cls = self._resolve_model_class()

        print(f"[{self.name}] Loading processor for '{self.hf_model_id}'"
              f"{f' with {self.processor_kwargs}' if self.processor_kwargs else ''} …")
        self._processor = AutoProcessor.from_pretrained(
            self.hf_model_id,
            token=self.hf_token,
            cache_dir=self.hf_cache,
            trust_remote_code=True,
            **self.processor_kwargs,
        )

        bnb = self._build_quantization_config()

        load_kwargs = {
            "token": self.hf_token,
            "cache_dir": self.hf_cache,
            "dtype": self.compute_dtype,
            "trust_remote_code": True,
            # accelerate places each weight as it loads, so host RAM never holds the whole
            # model, and "auto" spreads across every visible GPU — the only way one larger
            # than a single card loads. It caps all GPUs but the last, which then takes the
            # remainder, so splits skew; CUDA_VISIBLE_DEVICES=0 pins to one card.
            "device_map": "auto" if self.device == "cuda" else self.device,
            # No bitsandbytes config means a full-precision load: omit the key entirely.
            **({"quantization_config": bnb} if bnb is not None else {}),
        }

        print(
            f"[{self.name}] Loading model via {self.model_class} onto "
            f"{load_kwargs['device_map']} (dtype={self.compute_dtype}, "
            f"quantization={self.quantization_level}) …"
        )

        self._model = model_cls.from_pretrained(self.hf_model_id, **load_kwargs)
        self._model.eval()
        self._report_placement()

    def _report_placement(self) -> None:
        """Print where the weights landed, loudly if any of them left the GPU."""
        placement = getattr(self._model, "hf_device_map", None) or {"": self.device}
        devices = sorted({str(d) for d in placement.values()})
        print(f"[{self.name}] Model loaded on {', '.join(devices)} | "
              f"dtype: {next(self._model.parameters()).dtype}")

        if self.device == "cuda":
            # Live tensors only: mem_get_info would add the allocator's retained blocks,
            # which vary per card with the work it did and so exaggerate an uneven split.
            print(f"[{self.name}] Resident: " + " | ".join(
                f"cuda:{i} {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB"
                for i in range(torch.cuda.device_count())))

        offloaded = [d for d in devices if d in ("cpu", "disk")]
        if offloaded and self.device == "cuda":
            # Offloading is silent: generation runs orders of magnitude slower, not fails.
            print(f"[{self.name}] WARNING: no GPU room for the whole model — part of it "
                  f"sits on {', '.join(offloaded)} and generation will crawl. Use "
                  f"quantization_level='4bit', fewer frames, or a larger GPU.")
        print()

    def run(self, request: InferenceRequest) -> dict:
        """Generate text output using local Transformers model."""
        self._ensure_loaded()

        # Some models (e.g. Qwen VL) expect images embedded directly in content entries.
        # Others (e.g. Gemma3) want a plain placeholder and images passed separately
        # through the processor's `images` kwarg. Controlled by _INLINE_IMAGE_MODEL_PATTERNS.
        uses_inline_images = any(p in self.hf_model_id.lower() for p in _INLINE_IMAGE_MODEL_PATTERNS)
        all_pil_images: list[Image.Image] = []
        content_list: list[dict] = []

        for block in request.content:
            if isinstance(block, TextBlock):
                content_list.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                pil = block.load()
                if uses_inline_images:
                    content_list.append({"type": "image", "image": pil})
                else:
                    all_pil_images.append(pil)
                    content_list.append({"type": "image"})

        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": content_list})

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs: dict = {"text": [text], "return_tensors": "pt"}
        if all_pil_images:
            processor_kwargs["images"] = all_pil_images
        inputs = self._processor(**processor_kwargs).to(self.device)

        generation_kwargs = {
            "do_sample": request.do_sample,
            "temperature": request.temperature if request.do_sample else None,
            "top_p": request.top_p if request.do_sample else None,
        }
        if request.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = request.max_new_tokens
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        new_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        return {"text": self._processor.batch_decode(new_ids, skip_special_tokens=True)[0], "logs": []}
