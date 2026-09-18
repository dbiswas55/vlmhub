"""
One script: builds a Model, sends a request, checks a real response comes
back. This is the usage example too — read it top to bottom for the whole API.

    python tests/test_generate.py
    python tests/test_generate.py --client ollama/gemma3-4b
    python tests/test_generate.py --client transformers/gemma3-4b --image path/to/img.png
"""

import argparse

from vlmhub import Model, TextBlock, ImageBlock


def test_generate(client_name: str = "", image_path: str | None = None) -> None:
    model = Model(client_name)
    model.report()

    content = [TextBlock("Say hello in one short sentence.")]
    if image_path:
        content = [ImageBlock(image_path=image_path), TextBlock("What does this image show?")]

    response = model.generate(content)
    assert response["text"].strip(), "Expected non-empty generation"
    print(response["text"])
    for line in response.get("logs", []):
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="", help="'hosting/model'; empty uses models.json's active client")
    ap.add_argument("--image", default=None, help="optional image to send with the prompt")
    args = ap.parse_args()
    test_generate(args.client, args.image)
