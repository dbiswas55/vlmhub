"""
test_vqa.py
============
Downloads a few real image+questions+answers samples from the VQAv2 dataset
(lmms-lab-encoder/VQAv2 on the HF Hub) into <out_dir>/sample_XX/, skipping
any sample already downloaded, then asks vlmhub's Model to answer each
question and prints it next to the real ground-truth answer.

Each sample is its own subfolder:
    vqa_samples/
      sample_01/
        image.jpg
        qa.json   # {"questions": [{"question": ..., "answer": ...}, ...]}
      sample_02/
        ...

Usage
-----
    python tests/test_vqa.py
    python tests/test_vqa.py --sample input/vqa_samples --client ollama/gemma3-4b --n 3
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
QUESTIONS_PER_SAMPLE = 3


def ensure_samples(out_dir: Path, n: int = 3, q_per_sample: int = QUESTIONS_PER_SAMPLE) -> list[dict]:
    """Make sure n real VQAv2 samples (each an image with q_per_sample
    question+answer pairs) exist locally as out_dir/sample_XX/{image.jpg,qa.json},
    downloading only if they aren't already there. Returns a list of
    {"dir": Path, "image_path": Path, "questions": [{"question", "answer"}, ...]}."""
    width = max(2, len(str(n)))
    sample_dirs = [out_dir / f"sample_{i:0{width}d}" for i in range(1, n + 1)]

    # --- reuse what's already on disk, if complete ---
    def load_cached() -> list[dict] | None:
        samples = []
        for sample_dir in sample_dirs:
            image_path = sample_dir / "image.jpg"
            qa_path = sample_dir / "qa.json"
            if not (image_path.exists() and qa_path.exists()):
                return None
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            samples.append({"dir": sample_dir, "image_path": image_path, "questions": qa["questions"]})
        return samples

    if (cached := load_cached()) is not None:
        print(f"already have {n} samples in {out_dir}, skipping download")
        return cached

    # --- download the shard and pick n images with enough questions ---
    shard_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    rows = pq.ParquetFile(shard_path).read_row_group(0).to_pylist()

    # Group rows by image_id (the dataset stores several questions per image,
    # all sharing the same image_id), preserving first-seen order, and keep
    # only images that have at least q_per_sample questions.
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["image_id"], []).append(row)

    chosen_groups = [g for g in groups.values() if len(g) >= q_per_sample][:n]

    # --- save each chosen image + its questions to its own subfolder ---
    samples = []
    for sample_dir, group in zip(sample_dirs, chosen_groups):
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / "image.jpg"
        image = Image.open(io.BytesIO(group[0]["image"]["bytes"])).convert("RGB")
        image.save(image_path)

        questions = [
            {"question": row["question"], "answer": row["multiple_choice_answer"]}
            for row in group[:q_per_sample]
        ]
        (sample_dir / "qa.json").write_text(json.dumps({"questions": questions}, indent=2), encoding="utf-8")

        print(f"downloaded {sample_dir} — {len(questions)} question(s)")
        samples.append({"dir": sample_dir, "image_path": image_path, "questions": questions})

    return samples


def run(sample_dir: Path, client_name: str = "", n: int = 3,
        models_path: Path | None = None) -> None:
    samples = ensure_samples(sample_dir, n)

    model = Model(client_name, models_path=models_path)
    model.report()

    # --- ask the model each question and compare to ground truth ---
    for sample in samples:
        print(f"\n[{sample['dir'].name}]")
        for qa in sample["questions"]:
            content = [ImageBlock(image_path=str(sample["image_path"])), TextBlock(qa["question"])]
            response = model.generate(content)
            print(f"  question  : {qa['question']}")
            print(f"  answer    : {qa['answer']}")
            print(f"  generated : {response['text'].strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="input/vqa_samples", type=Path)
    ap.add_argument("--client", default="", help='e.g. "ollama/gemma3-4b"; blank uses models.json\'s "active" client')
    ap.add_argument("--n", default=3, type=int)
    ap.add_argument("--models-path", type=Path, default=None, help="registry override JSON, merged over the bundled one")
    args = ap.parse_args()
    run(args.sample, args.client, args.n, args.models_path)
