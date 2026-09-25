"""
test_captioning.py
====================
Downloads a few real image+captions samples from the COCO Captions dataset
(jxie/coco_captions on the HF Hub) into <out_dir>/sample_XX/, skipping any
sample already downloaded, then asks vlmhub's Model to summarize each image
and prints it next to the human-written reference captions.

Each sample is its own subfolder:
    coco_samples/
      sample_01/
        image.jpg
        captions.json   # {"captions": ["...", "...", ...]}
      sample_02/
        ...

Usage
-----
    python tests/test_captioning.py
    python tests/test_captioning.py --sample input/coco_samples --client ollama/gemma3-4b --n 3
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


def ensure_samples(out_dir: Path, n: int = 3) -> list[dict]:
    """Make sure n real COCO samples (each an image with all its reference
    captions) exist locally as out_dir/sample_XX/{image.jpg,captions.json},
    downloading only if they aren't already there. Returns a list of
    {"dir": Path, "image_path": Path, "captions": list[str]}."""
    width = max(2, len(str(n)))
    sample_dirs = [out_dir / f"sample_{i:0{width}d}" for i in range(1, n + 1)]

    # --- reuse what's already on disk, if complete ---
    def load_cached() -> list[dict] | None:
        samples = []
        for sample_dir in sample_dirs:
            image_path = sample_dir / "image.jpg"
            captions_path = sample_dir / "captions.json"
            if not (image_path.exists() and captions_path.exists()):
                return None
            captions = json.loads(captions_path.read_text(encoding="utf-8"))["captions"]
            samples.append({"dir": sample_dir, "image_path": image_path, "captions": captions})
        return samples

    if (cached := load_cached()) is not None:
        print(f"already have {n} samples in {out_dir}, skipping download")
        return cached

    # --- download the shard and pick n images ---
    shard_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    rows = pq.ParquetFile(shard_path).read_row_group(0).to_pylist()

    # Group rows by cocoid (the dataset stores 5 caption rows per image, one
    # row per human caption, all sharing the same cocoid), preserving
    # first-seen order, so each sample gets all of its reference captions.
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["cocoid"], []).append(row)

    chosen_groups = list(groups.values())[:n]

    # --- save each chosen image + its captions to its own subfolder ---
    samples = []
    for sample_dir, group in zip(sample_dirs, chosen_groups):
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / "image.jpg"
        image = Image.open(io.BytesIO(group[0]["image"]["bytes"])).convert("RGB")
        image.save(image_path)

        captions = [row["caption"].strip() for row in group]
        (sample_dir / "captions.json").write_text(json.dumps({"captions": captions}, indent=2), encoding="utf-8")

        print(f"downloaded {sample_dir} — {len(captions)} caption(s)")
        samples.append({"dir": sample_dir, "image_path": image_path, "captions": captions})

    return samples


def run(sample_dir: Path, client_name: str = "", n: int = 3,
        models_path: Path | None = None) -> None:
    samples = ensure_samples(sample_dir, n)

    model = Model(client_name, models_path=models_path)
    model.report()

    # --- ask the model to summarize each image and compare to references ---
    for sample in samples:
        content = [
            ImageBlock(image_path=str(sample["image_path"])),
            TextBlock("Write a one-sentence summary of this image."),
        ]
        response = model.generate(content)
        print(f"\n[{sample['dir'].name}]")
        for i, caption in enumerate(sample["captions"], start=1):
            print(f"  reference {i}: {caption}")
        print(f"  generated  : {response['text'].strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="input/coco_samples", type=Path)
    ap.add_argument("--client", default="", help='e.g. "ollama/gemma3-4b"; blank uses models.json\'s "active" client')
    ap.add_argument("--n", default=3, type=int)
    ap.add_argument("--models-path", type=Path, default=None, help="registry override JSON, merged over the bundled one")
    args = ap.parse_args()
    run(args.sample, args.client, args.n, args.models_path)
