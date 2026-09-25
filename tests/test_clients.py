"""
test_clients.py
================
Smoke-tests a list of clients: sends each one the same COCO image (reusing
test_captioning's samples) with a one-sentence summary prompt, and reports
which clients work, the model that served them, and the params actually sent.

Usage
-----
    python tests/test_clients.py
    python tests/test_clients.py --clients anthropic/haiku-4.5 vertex_ai/flash-3.5 --temperature 0
    python tests/test_clients.py --clients vllm/qwen3vl-4b --models-path my_models.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vlmhub import Model, TextBlock, ImageBlock
from test_captioning import ensure_samples

CLIENTS = [
    "vertex_ai/flash-3.5lite",
    "openai/gpt-4o-mini",
    "anthropic/haiku-4.5",

    # "vertex_ai/flash-3.8",
    # "vertex_ai/pro-2.5",

    # "openai/gpt-4o",
    # "openai/gpt-4.1",
    # "openai/gpt-5.4",

    # "anthropic/sonnet-4.6",
    # "anthropic/sonnet-5",

    # "gemini/gemma4-26b-a4b",

    # "openrouter/haiku-4.5",

    # "ollama/gemma3-4b",
]
PROMPT = "Write a one-sentence summary of this image."


def run(sample_dir: Path, clients: list[str], temperature: float | None = None,
        models_path: Path | None = None) -> None:
    image_path = ensure_samples(sample_dir, n=1)[0]["image_path"]
    content = [ImageBlock(image_path=str(image_path)), TextBlock(PROMPT)]
    overrides = {} if temperature is None else {"temperature": temperature}

    results = {}
    for name in clients:
        start = time.perf_counter()
        try:
            response = Model(name, models_path=models_path).generate(content, **overrides)
        except Exception as e:
            results[name] = False
            print(f"\n{name:22} FAIL {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            continue
        results[name] = True
        dropped = [log for log in response["logs"] if log.startswith("DROPPED")]
        print(f"\n{name:22} OK   {time.perf_counter() - start:.1f}s  "
              f"{response['model']}  {response['params']}  {dropped or ''}")
        print(f"{'':22} {(response['text'] or '').strip()[:100]!r}")

    ok = sum(results.values())
    print(f"\nSummary: {ok} OK, {len(results) - ok} FAIL")
    for name, passed in results.items():
        print(f"  {'OK  ' if passed else 'FAIL'} {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="input/coco_samples", type=Path)
    ap.add_argument("--clients", nargs="+", default=CLIENTS, help='e.g. "anthropic/opus-5.5 openai/gpt-4o"')
    ap.add_argument("--temperature", type=float, default=None, help="blank uses the registry default")
    ap.add_argument("--models-path", type=Path, default=None, help="registry override JSON, merged over the bundled one")
    args = ap.parse_args()
    run(args.sample, args.clients, args.temperature, args.models_path)
