"""
test_judge.py
==============
Checks a model as an LLM judge: asks it to grade a few built-in candidate
answers of known quality on a 1-5 rubric, and reports whether each score lands
in the expected range. Text only, so any model can run it, including judges
that take no images (prometheus-7b, gpt-oss-20b, ...). The prompt uses
Prometheus' absolute-grading format ("Feedback: ... [RESULT] <1-5>").

Usage
-----
    python tests/test_judge.py
    python tests/test_judge.py --client vllm/prometheus-7b --models-path my_models.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from vlmhub import Model, TextBlock

SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria."

RUBRIC = """[Is the response correct and complete compared to the reference answer?]
Score 1: The response is incorrect or contradicts the reference answer.
Score 2: The response is mostly incorrect, with only a minor correct element.
Score 3: The response is partially correct but misses or gets wrong an important part.
Score 4: The response is correct, with only a minor omission or imprecision.
Score 5: The response is fully correct and complete."""

PROMPT = """###Task Description:
An instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric are given.
1. Write detailed feedback that assesses the quality of the response strictly based on the score rubric.
2. After the feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: "Feedback: (feedback) [RESULT] (an integer between 1 and 5)"
4. Do not generate any other opening, closing, or explanations.

###The instruction to evaluate:
{question}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Feedback:"""

# (question, reference, candidate, expected score range)
CASES = [
    ("At what temperature does water boil at sea level, in degrees Celsius?",
     "100 degrees Celsius.",
     "Water boils at 100 °C at sea level.",
     (4, 5)),
    ("Who wrote the novel 'Pride and Prejudice'?",
     "Jane Austen.",
     "It was written by Charlotte Brontë in 1813.",
     (1, 2)),
    ("Name the three primary colors of light.",
     "Red, green and blue.",
     "Red and blue.",
     (2, 3)),
    ("What is 15 multiplied by 12?",
     "180.",
     "15 × 12 = 170.",
     (1, 2)),
]


def parse_score(text: str) -> int | None:
    """The last "[RESULT] n" (or "Score: n") in the judge's output, if any."""
    matches = re.findall(r"(?:\[RESULT\]|Score)\s*:?\s*\(?([1-5])\b", text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def run(client_name: str = "", models_path: Path | None = None, temperature: float = 0.0) -> None:
    model = Model(client_name, models_path=models_path)
    model.report()

    # --- ask the model to grade each candidate answer ---
    passed = 0
    for i, (question, reference, candidate, (low, high)) in enumerate(CASES, start=1):
        prompt = PROMPT.format(question=question, response=candidate, reference=reference, rubric=RUBRIC)
        response = model.generate([TextBlock(prompt)], system_prompt=SYSTEM_PROMPT, temperature=temperature)
        text = (response["text"] or "").strip()
        score = parse_score(text)
        ok = score is not None and low <= score <= high
        passed += ok

        print(f"\n[case {i}] {'OK  ' if ok else 'FAIL'} score={score} expected={low}-{high}")
        print(f"  question  : {question}")
        print(f"  candidate : {candidate}")
        print(f"  judge     : {text[:300]!r}")

    print(f"\nSummary: {passed}/{len(CASES)} scores in the expected range")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", default="", help='e.g. "vllm/prometheus-7b"; blank uses models.json\'s "active" client')
    ap.add_argument("--models-path", type=Path, default=None, help="registry override JSON, merged over the bundled one")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 (default) for repeatable scores")
    args = ap.parse_args()
    run(args.client, args.models_path, args.temperature)
