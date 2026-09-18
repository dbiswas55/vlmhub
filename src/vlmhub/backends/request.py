"""Data structures shared by all backends — what you send to a model."""

from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


@dataclass
class TextBlock:
    """A plain text segment in the prompt."""
    text: str


@dataclass
class ImageBlock:
    """An image in the prompt: a file on disk, or a decoded PIL image.

    Exactly one of image_path/image must be set. image (no file backing it)
    is encoded to PNG on demand -- lossless, and accepted natively by every
    backend that needs raw bytes (Gemini, OpenAI-compatible).
    """
    image_path: str | None = None
    image: Image.Image | None = None

    def __post_init__(self):
        if (self.image_path is None) == (self.image is None):
            raise ValueError("ImageBlock requires exactly one of image_path or image.")

    def load(self) -> Image.Image:
        if self.image is not None:
            return self.image.convert("RGB")
        with Image.open(self.image_path) as img:
            return img.convert("RGB")

    def read_bytes(self) -> bytes:
        if self.image is not None:
            buf = io.BytesIO()
            self.image.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
        with open(self.image_path, "rb") as f:
            return f.read()

    def mime_type(self) -> str:
        if self.image is not None:
            return "image/png"
        mime, _ = mimetypes.guess_type(self.image_path)
        return mime or "image/jpeg"

    def as_data_uri(self) -> str:
        b64 = base64.b64encode(self.read_bytes()).decode("utf-8")
        return f"data:{self.mime_type()};base64,{b64}"

    @property
    def label(self) -> str:
        """Short identifier for debug/logging output."""
        return Path(self.image_path).name if self.image_path is not None else "<in-memory image>"


ContentBlock = TextBlock | ImageBlock


@dataclass
class InferenceRequest:
    """Ordered list of text/image blocks — same structure for every backend.

    Usage:
        # Single image + instruction
        req = InferenceRequest(content=[
            ImageBlock(image_path="input/images/image_1.png"),
            TextBlock("What is the core concept shown here?"),
        ], system_prompt="", max_new_tokens=4096, temperature=1.0, top_p=1.0)

        # Multiple images + instruction
        req = InferenceRequest(content=[
            ImageBlock(image_path="input/images/image_1.png"),
            ImageBlock(image_path="input/images/image_2.png"),
            TextBlock("Compare the two slides."),
        ], system_prompt="", max_new_tokens=4096, temperature=1.0, top_p=1.0)

        # In-memory PIL image (e.g. a decoded video frame) -- no file needed
        req = InferenceRequest(content=[
            ImageBlock(image=frame),
            TextBlock("What is happening in this frame?"),
        ], system_prompt="", max_new_tokens=4096, temperature=1.0, top_p=1.0)
    """

    content: list[ContentBlock]
    system_prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0

    @property
    def has_images(self) -> bool:
        return any(isinstance(b, ImageBlock) for b in self.content)

    def image_blocks(self) -> list[ImageBlock]:
        return [b for b in self.content if isinstance(b, ImageBlock)]

    def to_openai_messages(self) -> list[dict]:
        """This request as an OpenAI-style messages list — what LiteLLM
        (and any OpenAI-compatible client) expects. Centralized here so
        every API-hosted backend builds it identically, once."""
        content: list[dict] = []
        for block in self.content:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                content.append({"type": "image_url", "image_url": {"url": block.as_data_uri()}})

        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages
