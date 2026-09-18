"""
test_captioning.py
====================
Downloads a few real image+caption pairs from the COCO Captions dataset
(jxie/coco_captions on the HF Hub) into input/<name>/, skipping any sample
already downloaded, then asks vlmhub's Model to summarize each image and
prints it next to the human-written reference caption.

Usage
-----
    python tests/test_captioning.py
    python tests/test_captioning.py --sample input/coco_sample --client ollama/gemma3-4b --n 3
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from vlmhub import Model, TextBlock, ImageBlock

DATASET_REPO = "jxie/coco_captions"
DATASET_FILE = "data/test-00000-of-00009-4a09b5aab8de74d3.parquet"


def ensure_samples(out_dir: Path, n: int = 3) -> list[tuple[Path, str]]:
    """Make sure n real COCO image+caption pairs exist locally, downloading
    only if they aren't already there. Returns (image_path, reference_caption)
    pairs, in order."""
    captions_path = out_dir / "captions.json"
    image_paths = [out_dir / f"image_{i}.jpg" for i in range(1, n + 1)]
    if captions_path.exists() and all(p.exists() for p in image_paths):
        print(f"already have {n} samples in {out_dir}, skipping download")
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
        return [(p, captions[p.name]) for p in image_paths]

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    # The dataset stores 5 caption rows per image (one row per human caption,
    # all sharing the same cocoid) — keep only the first row per image so the
    # n samples are n distinct images, not n captions of the same image.
    rows = pq.ParquetFile(shard_path).read_row_group(0).to_pylist()
    seen_cocoids = set()
    distinct_rows = []
    for row in rows:
        if row["cocoid"] in seen_cocoids:
            continue
        seen_cocoids.add(row["cocoid"])
        distinct_rows.append(row)
        if len(distinct_rows) == n:
            break

    samples = []
    captions = {}
    for image_path, row in zip(image_paths, distinct_rows):
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        image.save(image_path)
        reference = row["caption"].strip()
        print(f"downloaded {image_path} — reference: {reference}")
        samples.append((image_path, reference))
        captions[image_path.name] = reference

    captions_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")
    return samples


def run(sample_dir: Path, client_name: str = "", n: int = 3) -> None:
    samples = ensure_samples(sample_dir, n)

    model = Model(client_name)
    model.report()

    for image_path, reference in samples:
        content = [
            ImageBlock(image_path=str(image_path)),
            TextBlock("Write a one-sentence summary of this image."),
        ]
        response = model.generate(content)
        print(f"\n[{image_path.name}]")
        print(f"  reference : {reference}")
        print(f"  generated : {response['text'].strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="input/coco_sample", type=Path)
    ap.add_argument("--client", default="", help='e.g. "ollama/gemma3-4b"; blank uses models.json\'s "active" client')
    ap.add_argument("--n", default=3, type=int)
    args = ap.parse_args()
    run(args.sample, args.client, args.n)
