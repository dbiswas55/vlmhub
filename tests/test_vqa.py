"""
test_vqa.py
============
Downloads a few real image+question+answer triples from the VQAv2 dataset
(lmms-lab-encoder/VQAv2 on the HF Hub) into input/<name>/, skipping any
sample already downloaded, then asks vlmhub's Model to answer each question
and prints it next to the real ground-truth answer.

Usage
-----
    python tests/test_vqa.py
    python tests/test_vqa.py --sample input/vqa_sample --client ollama/gemma3-4b --n 3
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

DATASET_REPO = "lmms-lab-encoder/VQAv2"
DATASET_FILE = "data/validation-00000-of-00068.parquet"


def ensure_samples(out_dir: Path, n: int = 3) -> list[tuple[Path, str, str]]:
    """Make sure n real VQAv2 image+question+answer triples exist locally,
    downloading only if they aren't already there. Returns
    (image_path, question, ground_truth_answer) triples, in order."""
    qa_path = out_dir / "qa.json"
    image_paths = [out_dir / f"image_{i}.jpg" for i in range(1, n + 1)]
    if qa_path.exists() and all(p.exists() for p in image_paths):
        print(f"already have {n} samples in {out_dir}, skipping download")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        return [(p, qa[p.name]["question"], qa[p.name]["answer"]) for p in image_paths]

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    # The dataset stores several questions per image (all sharing the same
    # image_id) — keep only the first row per image so the n samples are n
    # distinct images, not n questions about the same image.
    rows = pq.ParquetFile(shard_path).read_row_group(0).to_pylist()
    seen_image_ids = set()
    distinct_rows = []
    for row in rows:
        if row["image_id"] in seen_image_ids:
            continue
        seen_image_ids.add(row["image_id"])
        distinct_rows.append(row)
        if len(distinct_rows) == n:
            break

    samples = []
    qa = {}
    for image_path, row in zip(image_paths, distinct_rows):
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        image.save(image_path)
        question = row["question"]
        answer = row["multiple_choice_answer"]
        print(f"downloaded {image_path} — Q: {question}  A: {answer}")
        samples.append((image_path, question, answer))
        qa[image_path.name] = {"question": question, "answer": answer}

    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return samples


def run(sample_dir: Path, client_name: str = "", n: int = 3) -> None:
    samples = ensure_samples(sample_dir, n)

    model = Model(client_name)
    model.report()

    for image_path, question, answer in samples:
        content = [ImageBlock(image_path=str(image_path)), TextBlock(question)]
        response = model.generate(content)
        print(f"\n[{image_path.name}]")
        print(f"  question  : {question}")
        print(f"  answer    : {answer}")
        print(f"  generated : {response['text'].strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="input/vqa_sample", type=Path)
    ap.add_argument("--client", default="", help='e.g. "ollama/gemma3-4b"; blank uses models.json\'s "active" client')
    ap.add_argument("--n", default=3, type=int)
    args = ap.parse_args()
    run(args.sample, args.client, args.n)
