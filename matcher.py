import anthropic
import base64
import json
from PIL import Image
import io

ANIMAL_EMOJI = {
    "lion": "🦁", "tiger": "🐯", "leopard": "🐆", "cheetah": "🐆", "jaguar": "🐆",
    "wolf": "🐺", "fox": "🦊", "coyote": "🐺", "husky": "🐶", "golden retriever": "🐶",
    "poodle": "🐩", "bulldog": "🐶", "german shepherd": "🐕", "labrador": "🐶",
    "chimpanzee": "🐒", "gorilla": "🦍", "orangutan": "🦧", "baboon": "🐒",
    "polar bear": "🐻‍❄️", "brown bear": "🐻", "giant panda": "🐼", "koala": "🐨",
    "horse": "🐴", "zebra": "🦓", "giraffe": "🦒", "elephant": "🐘", "rhinoceros": "🦏",
    "hippo": "🦛", "deer": "🦌", "moose": "🦌", "ram": "🐏", "goat": "🐐",
    "rabbit": "🐰", "hamster": "🐹", "squirrel": "🐿️", "raccoon": "🦝", "skunk": "🦨",
    "owl": "🦉", "eagle": "🦅", "parrot": "🦜", "penguin": "🐧", "flamingo": "🦩",
    "dolphin": "🐬", "seal": "🦭", "otter": "🦦", "cat": "🐱", "lynx": "🐱",
    "dog": "🐶", "bear": "🐻", "monkey": "🐒", "snake": "🐍", "crocodile": "🐊",
}

PROMPT = """Look at this photo of a person and determine which animal they most resemble based on their facial features, expressions, and overall appearance. Be playful and creative.

Respond ONLY with a valid JSON object in this exact format:
{
  "top_match": "animal name",
  "reason": "one fun sentence explaining why",
  "runners_up": [
    {"animal": "animal name", "score": 0.0},
    {"animal": "animal name", "score": 0.0},
    {"animal": "animal name", "score": 0.0},
    {"animal": "animal name", "score": 0.0}
  ]
}

Use lowercase animal names. Scores for runners_up should be between 0 and 1, with the top match implicitly at 1.0."""


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def match(image: Image.Image, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _image_to_base64(image),
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )

    data = json.loads(response.content[0].text)
    top = data["top_match"].lower()
    data["top_emoji"] = ANIMAL_EMOJI.get(top, "🐾")

    for r in data.get("runners_up", []):
        r["emoji"] = ANIMAL_EMOJI.get(r["animal"].lower(), "🐾")

    return data
