"""
generate_summary.py
====================
One-file demo: ensures a sample (2 real Commons diagrams + a source
description) exists locally, skipping any file already downloaded, then
generates a summary through vlmhub's Model.

Usage
-----
    python generate_summary.py
    python generate_summary.py --sample input/my_sample --client ollama/gemma3-4b --width 1536
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from vlmhub import Model, TextBlock, ImageBlock

# Verified real Commons files, CC BY-SA.
IMAGES = [
    "Multi-Layer_Neural_Network-Vector-Blank.svg",
    "Colored_neural_network.svg",
]

SOURCE_TEXT = (
    "Two diagrams of artificial neural networks. The first is a blank "
    "multi-layer network showing an input layer, one or more hidden "
    "layers, and an output layer connected by weighted edges, without "
    "example values filled in. The second is a colored fully connected "
    "network of the same layered structure, with each node's connections "
    "to every node in the adjacent layer visibly rendered."
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vlmhub-sample-downloader/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def ensure_sample(out_dir: Path, width: int = 1024) -> list[Path]:
    """Make sure the sample's images and source.txt exist locally, downloading
    only whatever is missing. Returns the image paths, in order."""
    image_paths = []
    for i, filename in enumerate(IMAGES, start=1):
        dest = out_dir / f"image_{i}_{Path(filename).stem}.png"
        image_paths.append(dest)
        if dest.exists():
            print(f"already have {dest}, skipping download")
            continue
        url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width={width}"
        print(f"downloading {url}")
        _download(url, dest)

    source_path = out_dir / "source.txt"
    if source_path.exists():
        print(f"already have {source_path}, skipping write")
    else:
        source_path.write_text(SOURCE_TEXT.strip() + "\n", encoding="utf-8")

    return image_paths


def generate_summary(sample_dir: Path, client_name: str = "", width: int = 1024) -> str:
    image_paths = ensure_sample(sample_dir, width)
    source_text = (sample_dir / "source.txt").read_text(encoding="utf-8").strip()

    model = Model(client_name)
    model.report()

    content = [ImageBlock(image_path=str(p)) for p in image_paths]
    content.append(TextBlock(
        f"Source description: {source_text}\n\n"
        "Write a two-sentence summary of what these diagrams show."
    ))

    response = model.generate(content)
    return response["text"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="input/test_sample", type=Path)
    ap.add_argument("--client", default="", help='e.g. "ollama/gemma3-4b"; blank uses models.json\'s "active" client')
    ap.add_argument("--width", default=1024, type=int)
    args = ap.parse_args()
    print(generate_summary(args.sample, args.client, args.width))
